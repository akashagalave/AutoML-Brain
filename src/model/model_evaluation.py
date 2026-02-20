import json
import logging
import joblib
import pandas as pd
import mlflow
import dagshub

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)

# Initialize DagsHub MLflow tracking
dagshub.init(
    repo_owner='akashagalaveaaa1',
    repo_name='AutoML-Brain',
    mlflow=True
)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


DATA_PATH = "data/processed/test_features.csv"
MODEL_DIR = Path("models/model_artifacts")
REPORT_DIR = Path("reports/evaluation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


model = joblib.load(MODEL_DIR / "model.pkl")
feature_columns = json.load(open(MODEL_DIR / "feature_columns.json"))
threshold = json.load(open(MODEL_DIR / "best_threshold.json"))["threshold"]

df = pd.read_csv(DATA_PATH)

TARGET = "Churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X[feature_columns]

y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

metrics = {
    "roc_auc": float(roc_auc_score(y, y_prob)),
    "pr_auc": float(average_precision_score(y, y_prob)),
    "accuracy": float(accuracy_score(y, y_pred)),
    "f1_score": float(f1_score(y, y_pred))
}

logger.info(metrics)

with open(REPORT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


mlflow.set_experiment("automl_brain_evaluation")

with mlflow.start_run(run_name="RandomForest_Optuna_Churn"):

    mlflow.log_metrics(metrics)
    mlflow.log_param("best_threshold", threshold)

    signature = mlflow.models.infer_signature(X, y_prob)

    mlflow.sklearn.log_model(
        model,
        name="churn_model",
        signature=signature,
        input_example=X.head(5)
    )

logger.info("Model evaluation completed successfully.")