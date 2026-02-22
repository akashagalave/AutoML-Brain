import time
from fastapi import FastAPI
from starlette.responses import Response
from prometheus_client import Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from .model_loader import load_model_assets
from .schema import PredictionRequest, PredictionResponse
from .config import APP_NAME, MODEL_VERSION
#from .cache import get_cached_prediction, set_cached_prediction
from src.common.feature_builder import build_feature_dataframe


app = FastAPI(title=APP_NAME)

# SLA-aware latency buckets
LATENCY_BUCKETS = (
    0.01, 0.02, 0.05,
    0.1, 0.2, 0.3,
    0.5, 1.0,
    float("inf"),
)

request_latency = Histogram(
    "churn_request_latency_seconds",
    "End-to-end request latency",
    ["path"],
    buckets=LATENCY_BUCKETS
)


@app.on_event("startup")
def preload():
    load_model_assets()


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    request_latency.labels(path=request.url.path).observe(duration)
    return response


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_dict = request.features.model_dump()

    # 1️⃣ Cache Check
    # cached = get_cached_prediction(input_dict)
    # if cached:
    #     return PredictionResponse(**cached)

    booster, threshold, feature_columns, feature_stats = load_model_assets()

    # 2️⃣ Build Feature DataFrame
    df = build_feature_dataframe(input_dict, feature_stats)

    # 3️⃣ Align exact feature order as training
    df = df[feature_columns]

    # 4️⃣ Restore categorical types
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    # 5️⃣ Native LightGBM Prediction
    prob = float(booster.predict(df)[0])
    prediction = prob >= threshold

    response_data = {
        "churn_probability": prob,
        "churn_risk_score": int(prob * 100),
        "will_churn": bool(prediction),
        "model_version": MODEL_VERSION,
        "top_reasons": None
    }

    #set_cached_prediction(input_dict, response_data)

    return PredictionResponse(**response_data)


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