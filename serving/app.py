import time
import os
import json
import numpy as np
from fastapi import FastAPI
from starlette.responses import Response
from fastapi.responses import ORJSONResponse
from prometheus_client import Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from .model_loader import load_model_assets
from .schema import PredictionRequest, PredictionResponse
from .config import APP_NAME, MODEL_VERSION
from src.common.feature_builder import build_feature_vector

# ==============================
# Optional Redis
# ==============================
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
redis_client = None

if REDIS_ENABLED:
    import redis
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )

# ==============================
# FastAPI App
# ==============================
app = FastAPI(
    title=APP_NAME,
    default_response_class=ORJSONResponse
)

# ==============================
# Prometheus Metrics
# ==============================
LATENCY_BUCKETS = (
    0.05, 0.1, 0.2,
    0.3, 0.5,
    1.0, 2.0, 3.0,
    5.0, 8.0, 10.0,
    float("inf"),
)

request_latency = Histogram(
    "churn_request_latency_seconds",
    "End-to-end request latency",
    ["path"],
    buckets=LATENCY_BUCKETS
)

# ==============================
# Global Model Assets
# ==============================
booster = None
threshold = None
feature_columns = None
feature_stats = None
categorical_columns = None
categorical_indices = None
category_maps = None
explainer = None


@app.on_event("startup")
def preload():
    global booster, threshold, feature_columns, feature_stats, \
           categorical_columns, categorical_indices, category_maps, explainer

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


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    request_latency.labels(path=request.url.path).observe(duration)
    return response


# ==============================
# FAST PREDICT
# ==============================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_dict = request.features.model_dump()

    vector = build_feature_vector(
        input_dict,
        feature_stats,
        feature_columns,
        categorical_columns,
        category_maps,
    )

    prob = float(
        booster.predict(
            vector,
            categorical_feature=categorical_indices
        )[0]
    )

    prediction = prob >= threshold

    return PredictionResponse(
        churn_probability=prob,
        churn_risk_score=int(prob * 100),
        will_churn=bool(prediction),
        model_version=MODEL_VERSION,
        top_reasons=None,
    )


# ==============================
# EXPLAIN ENDPOINT
# ==============================
@app.post("/predict/explain")
def predict_with_explain(request: PredictionRequest):

    input_dict = request.features.model_dump()

    vector = build_feature_vector(
        input_dict,
        feature_stats,
        feature_columns,
        categorical_columns,
        category_maps,
    )

    prob = float(
        booster.predict(
            vector,
            categorical_feature=categorical_indices
        )[0]
    )

    # SHAP values
    shap_values = explainer.shap_values(vector)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]

    # Top 3 impactful features
    abs_vals = np.abs(shap_values)
    top_indices = np.argsort(abs_vals)[-3:][::-1]

    explanations = []

    for idx in top_indices:
        feature = feature_columns[idx]
        impact = shap_values[idx]

        direction = "increased" if impact > 0 else "decreased"

        explanations.append({
            "feature": feature,
            "impact_value": float(impact),
            "explanation": f"{feature} {direction} the churn risk."
        })

    return {
        "churn_probability": prob,
        "will_churn": prob >= threshold,
        "model_version": MODEL_VERSION,
        "top_reasons": explanations
    }


# ==============================
# Health & Metrics
# ==============================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )