import pandas as pd
from pathlib import Path
from feature_engineering import build_features

ROOT = Path(__file__).parent.parent.parent

INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


train_df = pd.read_csv(INTERIM / "train_cleaned.csv")
train_features = build_features(train_df)
train_features.to_csv(PROCESSED / "train_features.csv", index=False)

val_df = pd.read_csv(INTERIM / "validation_cleaned.csv")
val_features = build_features(val_df)
val_features.to_csv(PROCESSED / "validation_features.csv", index=False)


test_df = pd.read_csv(INTERIM / "test_cleaned.csv")
test_features = build_features(test_df)
test_features.to_csv(PROCESSED / "test_features.csv", index=False)

print("FEATURE ENGINEERING COMPLETED (LightGBM Native Mode)")