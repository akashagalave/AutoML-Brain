import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

logger = logging.getLogger("model_benchmark")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

TRAIN_PATH = "data/processed/train_features.csv"
VAL_PATH = "data/processed/validation_features.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

TARGET = "Churn"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=300),
    "GradientBoosting": GradientBoostingClassifier(),
    "LightGBM": LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train)
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        eval_metric="logloss",
        use_label_encoder=False
    )
}

results = []

for name, model in models.items():
    logger.info(f"Training {name}")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    metrics = {
        "model": name,
        "roc_auc": roc_auc_score(y_val, y_prob),
        "pr_auc": average_precision_score(y_val, y_prob),
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    }

    results.append(metrics)

results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

print("\nMODEL COMPARISON")
print(results_df)