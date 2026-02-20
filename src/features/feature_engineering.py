import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --------------------------
# Define Columns
# --------------------------

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month",
]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]


TARGET_COL = "Churn"


# --------------------------
# Build Transformer
# --------------------------

def build_transformer():

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    return transformer


# --------------------------
# Feature Builder
# --------------------------

def build_features(df, mode="train", transformer=None):

    logger.info(f"Building features | mode={mode}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    if mode == "train":
        transformer = build_transformer()
        X_transformed = transformer.fit_transform(X)

        # Save transformer
        Path("models/feature_artifacts").mkdir(parents=True, exist_ok=True)
        joblib.dump(transformer, "models/feature_artifacts/feature_transformer.pkl")
        logger.info("Saved feature transformer artifact.")

    else:
        if transformer is None:
            transformer = joblib.load("models/feature_artifacts/feature_transformer.pkl")
        X_transformed = transformer.transform(X)

    X_transformed = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    )

    X_transformed[TARGET_COL] = y.values

    logger.info(f"Final feature shape: {X_transformed.shape}")

    return X_transformed, transformer