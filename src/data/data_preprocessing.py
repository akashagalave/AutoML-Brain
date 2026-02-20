import pandas as pd
import numpy as np
import logging
from pathlib import Path


# --------------------------
# Logging Setup
# --------------------------

logger = logging.getLogger("churn_preprocessing")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


# --------------------------
# Load Data
# --------------------------

def load_data(path: Path):
    df = pd.read_csv(path)
    logger.info(f"Loaded {path} | Shape = {df.shape}")
    return df


# --------------------------
# Remove PII
# --------------------------

def drop_pii(df):
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Removed PII column: customerID")
    return df


# --------------------------
# Type Fixing
# --------------------------

def fix_types(df):

    # Convert TotalCharges safely
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert SeniorCitizen to int8
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")

    logger.info("Fixed numeric datatypes.")
    return df


# --------------------------
# Target Encoding
# --------------------------

def process_target(df):

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")

    logger.info("Converted Churn target to binary.")
    return df


# --------------------------
# Feature Engineering
# --------------------------

def create_features(df):

    # Tenure buckets (lifecycle simulation)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1y", "1-2y", "2-4y", "4-6y"]
    )

    # Monthly charge intensity
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    logger.info("Created derived features.")
    return df


# --------------------------
# Handle Missing Categorical Values
# --------------------------

def handle_missing(df):

    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    logger.info("Handled missing categorical values.")
    return df


# --------------------------
# Save
# --------------------------

def save(path, df):
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned file: {path}")


# --------------------------
# Pipeline Runner
# --------------------------

def process_file(input_path, output_path):

    df = load_data(input_path)
    df = drop_pii(df)
    df = fix_types(df)
    df = process_target(df)
    df = create_features(df)
    df = handle_missing(df)
    df = df.drop_duplicates()

    save(output_path, df)


# --------------------------
# Main
# --------------------------

if __name__ == "__main__":

    ROOT = Path(__file__).parent.parent.parent

    raw_dir = ROOT / "data" / "raw"
    interim_dir = ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    process_file(raw_dir / "train.csv", interim_dir / "train_cleaned.csv")
    process_file(raw_dir / "validation.csv", interim_dir / "validation_cleaned.csv")
    process_file(raw_dir / "test.csv", interim_dir / "test_cleaned.csv")

    logger.info("CHURN DATA PREPROCESSING COMPLETED SUCCESSFULLY")