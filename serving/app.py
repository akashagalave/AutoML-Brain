import time
import os
import json
from fastapi import FastAPI
from starlette.responses import Response
from fastapi.responses import ORJSONResponse
from prometheus_client import Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from .model_loader import load_model_assets
from .schema import PredictionRequest, PredictionResponse
from .config import APP_NAME, MODEL_VERSION
from src.common.feature_builder import build_feature_dataframe

# Optional Redis
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
redis_client = None

if REDIS_ENABLED:
    import redis
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )

app = FastAPI(
    title=APP_NAME,
    default_response_class=ORJSONResponse
)

# SLA-aware latency buckets (honest)
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

# Global model objects
booster = None
threshold = None
feature_columns = None
feature_stats = None


@app.on_event("startup")
def preload():
    global booster, threshold, feature_columns, feature_stats
    booster, threshold, feature_columns, feature_stats = load_model_assets()


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

    # 🔥 Redis Cache Check
    if REDIS_ENABLED and redis_client:
        cache_key = f"churn:{json.dumps(input_dict, sort_keys=True)}"
        cached = redis_client.get(cache_key)
        if cached:
            return PredictionResponse(**json.loads(cached))

    # 1️⃣ Build Feature DataFrame
    df = build_feature_dataframe(input_dict, feature_stats)

    # 2️⃣ Align exact feature order
    df = df[feature_columns]

    # 3️⃣ Restore categorical types
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    # 4️⃣ LightGBM Prediction
    prob = float(booster.predict(df)[0])
    prediction = prob >= threshold

    response_data = {
        "churn_probability": prob,
        "churn_risk_score": int(prob * 100),
        "will_churn": bool(prediction),
        "model_version": MODEL_VERSION,
        "top_reasons": None
    }

    # 🔥 Store in Redis
    if REDIS_ENABLED and redis_client:
        redis_client.setex(
            cache_key,
            300,  # 5 min TTL
            json.dumps(response_data)
        )

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
