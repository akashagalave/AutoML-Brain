import pandas as pd
from pathlib import Path
from feature_engineering import build_features
import joblib


ROOT = Path(__file__).parent.parent.parent

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# --------------------------
# Train
# --------------------------

train_df = pd.read_csv(INTERIM / "train_cleaned.csv")
train_features, transformer = build_features(train_df, mode="train")
train_features.to_csv(PROCESSED / "train_features.csv", index=False)


# --------------------------
# Validation
# --------------------------

val_df = pd.read_csv(INTERIM / "validation_cleaned.csv")
val_features, _ = build_features(val_df, mode="inference", transformer=transformer)
val_features.to_csv(PROCESSED / "validation_features.csv", index=False)


# --------------------------
# Test
# --------------------------

test_df = pd.read_csv(INTERIM / "test_cleaned.csv")
test_features, _ = build_features(test_df, mode="inference", transformer=transformer)
test_features.to_csv(PROCESSED / "test_features.csv", index=False)

print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")