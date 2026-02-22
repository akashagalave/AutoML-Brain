import pandas as pd


def _tenure_bucket_fast(t):
    """
    Faster replacement for pd.cut.
    Reduces per-request CPU overhead.
    """
    if t <= 12:
        return "0-1y"
    elif t <= 24:
        return "1-2y"
    elif t <= 48:
        return "2-4y"
    else:
        return "4-6y"


def build_feature_dataframe(raw: dict, feature_stats: dict) -> pd.DataFrame:
    """
    Enterprise-safe feature builder.
    Must stay perfectly aligned with training preprocessing.
    Optimized for low-latency inference.
    """

    df = pd.DataFrame([raw])

    # ==============================
    # TYPE FIXES
    # ==============================

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")

    # ==============================
    # FAST TENURE BUCKET
    # ==============================

    df["tenure_bucket"] = df["tenure"].apply(_tenure_bucket_fast)

    # ==============================
    # DERIVED FEATURES
    # ==============================

    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["clv_proxy"] = df["MonthlyCharges"] * df["tenure"]
    df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["service_density"] = df["num_services"] / len(service_cols)

    # ==============================
    # CONTRACT STRENGTH
    # ==============================

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    df["contract_strength"] = df["Contract"].map(contract_map)

    # ==============================
    # PRECOMPUTED STAT USAGE
    # ==============================

    monthly_75 = feature_stats["monthly_charge_75th"]

    df["high_monthly_charge"] = (
        df["MonthlyCharges"] > monthly_75
    ).astype(int)

    df["is_fiber"] = (
        df["InternetService"] == "Fiber optic"
    ).astype(int)

    df["is_electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    df["tenure_contract_interaction"] = (
        df["tenure"] * df["contract_strength"]
    )

    df["high_charge_no_security"] = (
        (df["high_monthly_charge"] == 1)
        & (df["OnlineSecurity"] == "No")
    ).astype(int)

    return df