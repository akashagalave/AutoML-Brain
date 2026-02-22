import pandas as pd


def _tenure_bucket_fast(t: int) -> str:
    if t <= 12:
        return "0-1y"
    elif t <= 24:
        return "1-2y"
    elif t <= 48:
        return "2-4y"
    else:
        return "4-6y"


def build_feature_dataframe(
    raw: dict,
    feature_stats: dict,
    feature_columns: list,
    categorical_columns: list
) -> pd.DataFrame:
    """
    Ultra-optimized inference feature builder.
    Fully aligned with training pipeline.
    Avoids dtype scanning and unnecessary operations.
    """

    # =========================
    # BASIC TYPE FIXES
    # =========================

    raw["TotalCharges"] = float(raw.get("TotalCharges", 0) or 0)
    raw["SeniorCitizen"] = int(raw.get("SeniorCitizen", 0))

    tenure = int(raw["tenure"])
    monthly = float(raw["MonthlyCharges"])
    total = raw["TotalCharges"]

    # =========================
    # FAST DERIVED FEATURES
    # =========================

    raw["tenure_bucket"] = _tenure_bucket_fast(tenure)

    raw["charges_per_month"] = total / (tenure + 1)
    raw["clv_proxy"] = monthly * tenure
    raw["avg_charge_per_tenure"] = total / (tenure + 1)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    num_services = sum(
        1 for col in service_cols if raw.get(col) == "Yes"
    )

    raw["num_services"] = num_services
    raw["service_density"] = num_services / len(service_cols)

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    raw["contract_strength"] = contract_map.get(raw["Contract"], 0)

    monthly_75 = feature_stats["monthly_charge_75th"]

    raw["high_monthly_charge"] = int(monthly > monthly_75)
    raw["is_fiber"] = int(raw["InternetService"] == "Fiber optic")
    raw["is_electronic_check"] = int(
        raw["PaymentMethod"] == "Electronic check"
    )

    raw["tenure_contract_interaction"] = (
        tenure * raw["contract_strength"]
    )

    raw["high_charge_no_security"] = int(
        raw["high_monthly_charge"] == 1
        and raw.get("OnlineSecurity") == "No"
    )

    # =========================
    # BUILD ORDERED ROW
    # =========================

    ordered_row = [raw[col] for col in feature_columns]

    df = pd.DataFrame([ordered_row], columns=feature_columns)

    # Explicit categorical casting (no scanning)
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df