import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "charges_per_month", "clv_proxy",
    "avg_charge_per_tenure", "num_services",
    "service_density", "contract_strength",
    "tenure_contract_interaction",
    "is_fiber", "high_monthly_charge",
    "is_electronic_check", "high_charge_no_security"
]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents",
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "tenure_bucket"
]

TARGET_COL = "Churn"


def build_transformer():

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )

    return transformer


def build_features(df, mode="train", transformer=None):

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    if mode == "train":
        transformer = build_transformer()
        X_transformed = transformer.fit_transform(X)

        Path("models/feature_artifacts").mkdir(parents=True, exist_ok=True)
        joblib.dump(transformer, "models/feature_artifacts/feature_transformer.pkl")

    else:
        transformer = joblib.load("models/feature_artifacts/feature_transformer.pkl")
        X_transformed = transformer.transform(X)

    X_transformed = pd.DataFrame(X_transformed)
    X_transformed[TARGET_COL] = y.values

    return X_transformed, transformer