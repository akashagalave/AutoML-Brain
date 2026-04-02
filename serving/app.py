import os
import time
import json
import logging
import asyncio
import numpy as np

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from starlette.requests import Request
from starlette.responses import Response
from fastapi.responses import ORJSONResponse
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .model_loader import load_model_assets
from .schema import PredictionRequest, PredictionResponse
from .config import APP_NAME, MODEL_VERSION
from src.common.feature_builder import build_feature_vector


CANARY_SHADOW_URL = os.getenv(
    "CANARY_SHADOW_URL",
    "http://inference-canary-service.automl-brain.svc.cluster.local:80",
)
SHADOW_ENABLED = bool(CANARY_SHADOW_URL)
SHADOW_TIMEOUT = float(os.getenv("SHADOW_TIMEOUT_SECONDS", "2.0"))


app = FastAPI(title=APP_NAME, default_response_class=ORJSONResponse)
logger = logging.getLogger("automl-brain")


LATENCY_BUCKETS = (
    0.010, 0.025, 0.050,
    0.100, 0.200, 0.300, 0.350,
    0.500, 0.750,
    1.0, 1.5, 2.0,
    3.0, 5.0, float("inf"),
)

request_latency = Histogram(
    "churn_request_latency_seconds",
    "End-to-end request latency",
    ["path"],
    buckets=LATENCY_BUCKETS,
)

predictions_total = Counter(
    "churn_predictions_total",
    "Total churn predictions made",
    ["will_churn", "model_version"],
)

churn_positive_total = Counter(
    "churn_positive_predictions_total",
    "Predictions where will_churn=True",
    ["model_version"],
)

churn_negative_total = Counter(
    "churn_negative_predictions_total",
    "Predictions where will_churn=False",
    ["model_version"],
)

churn_probability_histogram = Histogram(
    "churn_probability_distribution",
    "Distribution of raw churn probability scores",
    ["model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


shadow_mirror_total = Counter(
    "churn_shadow_mirror_total",
    "Total requests mirrored to canary for shadow validation",
    ["status"],  
)


booster          = None
threshold        = None
feature_columns  = None
feature_stats    = None
categorical_columns  = None
categorical_indices  = None
category_maps    = None
explainer        = None


@app.on_event("startup")
def preload():
    global booster, threshold, feature_columns, feature_stats, \
           categorical_columns, categorical_indices, category_maps, explainer

    logger.info("Preloading model assets...")
    (
        booster,
        threshold,
        feature_columns,
        feature_stats,
        categorical_columns,
        categorical_indices,
        category_maps,
        explainer,
    ) = load_model_assets()
    logger.info(f"Model ready | version={MODEL_VERSION} | threshold={threshold:.4f}")
    logger.info(f"Shadow traffic enabled={SHADOW_ENABLED} | canary_url={CANARY_SHADOW_URL}")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    request_latency.labels(path=request.url.path).observe(time.perf_counter() - start)
    return response


async def _mirror_to_canary(payload: dict) -> None:
    """Async shadow call to canary /predict. Errors are swallowed intentionally."""
    if not SHADOW_ENABLED:
        shadow_mirror_total.labels(status="disabled").inc()
        return
    try:
        async with httpx.AsyncClient(timeout=SHADOW_TIMEOUT) as client:
            await client.post(f"{CANARY_SHADOW_URL}/predict", json=payload)
        shadow_mirror_total.labels(status="success").inc()
        logger.debug("Shadow mirror → canary: success")
    except Exception as exc:
        shadow_mirror_total.labels(status="error").inc()
        logger.debug(f"Shadow mirror → canary: failed ({exc})")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):

    input_dict = request.features.model_dump()

    def _infer():
        vector = build_feature_vector(
            input_dict,
            feature_stats,
            feature_columns,
            categorical_columns,
            category_maps,
        )
        prob = float(booster.predict(vector, categorical_feature=categorical_indices)[0])
        return prob

    try:
        prob = await run_in_threadpool(_infer)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    prediction = prob >= threshold

    predictions_total.labels(
        will_churn=str(bool(prediction)),
        model_version=MODEL_VERSION,
    ).inc()

    if prediction:
        churn_positive_total.labels(model_version=MODEL_VERSION).inc()
    else:
        churn_negative_total.labels(model_version=MODEL_VERSION).inc()

    churn_probability_histogram.labels(model_version=MODEL_VERSION).observe(prob)

    background_tasks.add_task(_mirror_to_canary, request.model_dump())

    return PredictionResponse(
        churn_probability=prob,
        churn_risk_score=int(prob * 100),
        will_churn=bool(prediction),
        model_version=MODEL_VERSION,
        top_reasons=None,
    )


@app.post("/predict/explain")
async def predict_with_explain(request: PredictionRequest):
    """
    Churn prediction with SHAP explanations.
    Slower path — use for investigation, not real-time throughput.
    """
    input_dict = request.features.model_dump()

    def _infer_with_shap():
        vector = build_feature_vector(
            input_dict,
            feature_stats,
            feature_columns,
            categorical_columns,
            category_maps,
        )
        prob = float(booster.predict(vector, categorical_feature=categorical_indices)[0])

        explanations = []
        if explainer is not None:
            sv = explainer.shap_values(vector)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]
            abs_vals   = np.abs(sv)
            top_indices = np.argsort(abs_vals)[-3:][::-1]
            for idx in top_indices:
                feature   = feature_columns[idx]
                impact    = sv[idx]
                direction = "increased" if impact > 0 else "decreased"
                explanations.append({
                    "feature":       feature,
                    "impact_value":  float(impact),
                    "explanation":   f"{feature} {direction} the churn risk.",
                })

        return prob, explanations

    try:
        prob, explanations = await run_in_threadpool(_infer_with_shap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    prediction = prob >= threshold
    predictions_total.labels(
        will_churn=str(bool(prediction)),
        model_version=MODEL_VERSION,
    ).inc()

    return {
        "churn_probability": prob,
        "will_churn":        bool(prediction),
        "model_version":     MODEL_VERSION,
        "top_reasons":       explanations,
    }


@app.get("/health")
def health():
    return {
        "status":                  "ok",
        "model_version":           MODEL_VERSION,
        "model_loaded":            booster is not None,
        "threshold":               threshold,
        "shadow_traffic_enabled":  SHADOW_ENABLED,
        "canary_url":              CANARY_SHADOW_URL if SHADOW_ENABLED else None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)