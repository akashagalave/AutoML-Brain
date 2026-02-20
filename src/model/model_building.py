import pandas as pd
import numpy as np
import logging
import json
import joblib
import shap
import matplotlib.pyplot as plt
import optuna

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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
# Optuna Objective
# --------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_auc = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1
    ).mean()

    return roc_auc


# --------------------------
# Run Optuna
# --------------------------

logger.info("Starting Optuna hyperparameter tuning...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

best_params = study.best_params
logger.info(f"Best Params: {best_params}")
logger.info(f"Best CV ROC-AUC: {study.best_value:.4f}")


# --------------------------
# Train Final Model
# --------------------------

best_params.update({
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
})

model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

logger.info("Final model trained.")


# --------------------------
# Threshold Optimization
# --------------------------

y_prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

best_threshold = thresholds[np.argmax(f1_scores)]
logger.info(f"Best threshold: {best_threshold:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_val, y_prob)
f1 = f1_score(y_val, y_pred)

logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")
logger.info(f"Validation F1: {f1:.4f}")


# --------------------------
# Save Artifacts
# --------------------------

joblib.dump(model, MODEL_DIR / "model.pkl")

with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(list(X_train.columns), f)

with open(MODEL_DIR / "best_threshold.json", "w") as f:
    json.dump({"threshold": float(best_threshold)}, f)

logger.info("Model artifacts saved.")

# --------------------------
# SHAP Explainability (Fixed for RandomForest)
# --------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# For binary classification, take positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# If 3D array (n_samples, n_features, 2)
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": X_val.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(REPORT_DIR / "shap_global_importance.csv", index=False)

shap.summary_plot(shap_values, X_val, show=False)
plt.savefig(REPORT_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.savefig(REPORT_DIR / "shap_summary_bar.png", bbox_inches="tight")
plt.close()

logger.info("SHAP reports generated.")