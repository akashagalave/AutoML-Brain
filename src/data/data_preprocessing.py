import pandas as pd
import numpy as np
import logging
from pathlib import Path


logger = logging.getLogger("churn_preprocessing")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_data(path: Path):
    df = pd.read_csv(path)
    logger.info(f"Loaded {path} | Shape = {df.shape}")
    return df


def drop_pii(df):
    return df.drop(columns=["customerID"], errors="ignore")


def fix_types(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")
    return df


def process_target(df):
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    return df


def create_features(df):

    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1y", "1-2y", "2-4y", "4-6y"]
    )

    df["clv_proxy"] = df["MonthlyCharges"] * df["tenure"]
    df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["service_density"] = df["num_services"] / len(service_cols)

    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    df["contract_strength"] = df["Contract"].map(contract_map)

    df["high_monthly_charge"] = (
        df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
    ).astype(int)

    df["is_electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    df["tenure_contract_interaction"] = (
        df["tenure"] * df["contract_strength"]
    )

    df["high_charge_no_security"] = (
        (df["high_monthly_charge"] == 1) &
        (df["OnlineSecurity"] == "No")
    ).astype(int)

    logger.info("Advanced features created.")
    return df


def handle_missing(df):
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    return df


def save(path, df):
    df.to_csv(path, index=False)
    logger.info(f"Saved {path}")


def process_file(input_path, output_path):

    df = load_data(input_path)
    df = drop_pii(df)
    df = fix_types(df)
    df = process_target(df)
    df = create_features(df)
    df = handle_missing(df)
    df = df.drop_duplicates()

    save(output_path, df)


if __name__ == "__main__":

    ROOT = Path(__file__).parent.parent.parent
    raw_dir = ROOT / "data" / "raw"
    interim_dir = ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    process_file(raw_dir / "train.csv", interim_dir / "train_cleaned.csv")
    process_file(raw_dir / "validation.csv", interim_dir / "validation_cleaned.csv")
    process_file(raw_dir / "test.csv", interim_dir / "test_cleaned.csv")

    logger.info("PREPROCESSING COMPLETED")