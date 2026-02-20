import pandas as pd
import numpy as np
import json
from pathlib import Path


def calculate_psi(expected, actual, bins=10):

    # Define bins from expected (train) distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi = np.sum((expected_perc - actual_perc) *
                 np.log(expected_perc / actual_perc))

    return psi


if __name__ == "__main__":

    TRAIN_PATH = "data/interim/train_cleaned.csv"
    TEST_PATH = "data/interim/test_cleaned.csv"

    REPORT_DIR = Path("reports/drift")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    drift_report = {}

    # Only numeric columns
    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if col == "Churn":
            continue

        psi_value = calculate_psi(train_df[col], test_df[col])
        drift_report[col] = float(psi_value)

    with open(REPORT_DIR / "psi_report.json", "w") as f:
        json.dump(drift_report, f, indent=4)

    print("Drift detection completed properly.")