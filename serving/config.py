from dotenv import load_dotenv
load_dotenv()

import os

# ---------------------------
# MLflow / DagsHub
# ---------------------------

MLFLOW_TRACKING_URI = "https://dagshub.com/akashagalaveaaa1/AutoML-Brain.mlflow"

MODEL_NAME = "churn_model"
MODEL_STAGE = "Production"

# ---------------------------
# Business Config
# ---------------------------

DEFAULT_THRESHOLD = 0.5

# ---------------------------
# Monitoring
# ---------------------------

APP_NAME = "ChurnGuard API"
MODEL_VERSION = os.getenv("MODEL_VERSION", "stable")

# ---------------------------
# Redis Config
# ---------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))