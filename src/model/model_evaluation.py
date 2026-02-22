import json
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import dagshub
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# =====================================================
# INIT MLFLOW
# =====================================================

dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="AutoML-Brain",
    mlflow=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_evaluation")


# =====================================================
# PATHS
# =====================================================

TEST_PATH = "data/processed/test_features.csv"
MODEL_DIR = Path("models/model_artifacts")
REPORT_DIR = Path("reports/evaluation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD TEST DATA
# =====================================================

logger.info("Loading test dataset...")
df = pd.read_csv(TEST_PATH)

TARGET = "Churn"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =====================================================
# LOAD FEATURE SCHEMA
# =====================================================

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    feature_columns = json.load(f)

X = X[feature_columns]


# =====================================================
# FIX CATEGORICAL TYPES
# =====================================================

categorical_cols = X.select_dtypes(include="object").columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype("category")

logger.info(f"Categorical columns restored: {categorical_cols}")


# =====================================================
# LOAD MODEL
# =====================================================

model_path = MODEL_DIR / "model.txt"
logger.info("Loading LightGBM native booster...")
model = lgb.Booster(model_file=str(model_path))


# =====================================================
# EVALUATION + MLFLOW LOGGING
# =====================================================

with mlflow.start_run() as run:

    logger.info("Running evaluation predictions...")
    y_prob = model.predict(X)

    with open(MODEL_DIR / "best_threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    metrics = {
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Log metrics
    mlflow.log_metrics(metrics)

    # 🔥 IMPORTANT: log model.txt as artifact manually
    mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")

    run_id = run.info.run_id

    # Correct URI for artifact-based model
    model_uri = f"runs:/{run_id}/model_artifacts/model.txt"

    # Save metrics report
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Create run_information.json
    run_information = {
        "run_id": run_id,
        "model_uri": model_uri,
        "model_type": "lightgbm_native",
        "threshold": float(threshold),
        "metrics": metrics
    }

    with open("run_information.json", "w") as f:
        json.dump(run_information, f, indent=4)

    logger.info("run_information.json created successfully.")
    logger.info("Evaluation + MLflow logging completed successfully.")