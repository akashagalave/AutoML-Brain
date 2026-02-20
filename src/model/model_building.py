import pandas as pd
import numpy as np
import logging
import json
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# --------------------------
# Logging
# --------------------------

logger = logging.getLogger("model_building")
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


# --------------------------
# Paths
# --------------------------

TRAIN_PATH = "data/processed/train_features.csv"
VAL_PATH = "data/processed/validation_features.csv"

MODEL_DIR = Path("models/model_artifacts")
REPORT_DIR = Path("reports/shap")
EVAL_DIR = Path("reports/evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# Load Data
# --------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

TARGET = "Churn"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]


# --------------------------
# Model Parameters
# --------------------------

params = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 64,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# --------------------------
# Train Model
# --------------------------

model = LGBMClassifier(**params)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc"
)

logger.info("Model training completed.")


# --------------------------
# Validation Metrics
# --------------------------

y_prob = model.predict_proba(X_val)[:, 1]
y_pred = model.predict(X_val)

roc_auc = roc_auc_score(y_val, y_prob)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

metrics = {
    "roc_auc": float(roc_auc),
    "accuracy": float(accuracy),
    "f1_score": float(f1),
}

logger.info(f"ROC-AUC: {roc_auc:.4f}")
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"F1-score: {f1:.4f}")

with open(EVAL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# --------------------------
# Save Model Artifacts
# --------------------------

joblib.dump(model, MODEL_DIR / "model.pkl")

with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(list(X_train.columns), f)

logger.info("Model artifacts saved.")


# --------------------------
# SHAP Explainability
# --------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_df = pd.DataFrame({
    "feature": X_val.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(REPORT_DIR / "shap_global_importance.csv", index=False)

shap.summary_plot(shap_values, X_val, show=False)
plt.savefig(REPORT_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.savefig(REPORT_DIR / "shap_summary_bar.png", bbox_inches="tight")
plt.close()

logger.info("SHAP explainability reports generated.")