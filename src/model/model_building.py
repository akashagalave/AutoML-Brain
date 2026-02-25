import pandas as pd
import numpy as np
import logging
import json
import shap
import matplotlib
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# =====================================================
# MATPLOTLIB HEADLESS
# =====================================================

matplotlib.use("Agg")

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_building")

# =====================================================
# PATHS
# =====================================================

TRAIN_PATH = "data/processed/train_features.csv"
VAL_PATH = "data/processed/validation_features.csv"

MODEL_DIR = Path("models/model_artifacts")
REPORT_DIR = Path("reports/shap")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

logger.info("Loading training and validation data...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

TARGET = "Churn"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

# =====================================================
# HANDLE CATEGORICALS
# =====================================================

categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

for col in categorical_cols:
    X_train[col] = X_train[col].astype("category")
    X_val[col] = X_val[col].astype("category")

logger.info(f"Categorical columns: {categorical_cols}")

# =====================================================
# OPTUNA OBJECTIVE
# =====================================================

def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "n_jobs": 1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            categorical_feature=categorical_cols
        )

        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y_train.iloc[val_idx], preds))

    return np.mean(scores)

# =====================================================
# RUN OPTUNA
# =====================================================

logger.info("Starting Optuna tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_jobs": 1,
    "random_state": 42
})

logger.info(f"Best Params: {best_params}")



model = lgb.LGBMClassifier(**best_params)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_cols
)

logger.info("Final model trained.")

y_prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

best_threshold = thresholds[np.argmax(f1_scores)]

roc_auc = roc_auc_score(y_val, y_prob)
f1 = f1_score(y_val, (y_prob >= best_threshold).astype(int))

logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")
logger.info(f"Validation F1: {f1:.4f}")
logger.info(f"Best Threshold: {best_threshold:.4f}")


model.booster_.save_model(str(MODEL_DIR / "model.txt"))

with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(list(X_train.columns), f)

with open(MODEL_DIR / "categorical_columns.json", "w") as f:
    json.dump(categorical_cols, f)

with open(MODEL_DIR / "best_threshold.json", "w") as f:
    json.dump({"threshold": float(best_threshold)}, f)


feature_stats = {
    "monthly_charge_75th": float(
        X_train["MonthlyCharges"].quantile(0.75)
    )
}

with open(MODEL_DIR / "feature_stats.json", "w") as f:
    json.dump(feature_stats, f)

logger.info("Model artifacts saved successfully.")


logger.info("Generating SHAP reports...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": X_val.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(REPORT_DIR / "shap_global_importance.csv", index=False)

shap.summary_plot(shap_values, X_val, show=False)
plt.tight_layout()
plt.savefig(REPORT_DIR / "shap_summary.png")
plt.close()

logger.info("SHAP reports generated successfully.")