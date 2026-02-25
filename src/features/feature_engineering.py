import pandas as pd
import logging

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


def build_features(df):

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X[TARGET_COL] = y

    logger.info("Native LightGBM features prepared (no sklearn transformer).")

    return X