import numpy as np


def _tenure_bucket_fast(t):
    if t <= 12:
        return "0-1y"
    elif t <= 24:
        return "1-2y"
    elif t <= 48:
        return "2-4y"
    else:
        return "4-6y"


def build_feature_vector(
    raw: dict,
    feature_stats: dict,
    feature_columns: list,
    categorical_columns: list,
    category_maps: dict,
):
    # ---- Derived features ----
    total_charges = float(raw["TotalCharges"])
    tenure = int(raw["tenure"])
    monthly = float(raw["MonthlyCharges"])

    charges_per_month = total_charges / (tenure + 1)
    clv_proxy = monthly * tenure
    avg_charge_per_tenure = total_charges / (tenure + 1)

    tenure_bucket = _tenure_bucket_fast(tenure)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    num_services = sum(
        1 for c in service_cols if raw[c] == "Yes"
    )
    service_density = num_services / len(service_cols)

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }

    contract_strength = contract_map[raw["Contract"]]

    high_monthly_charge = int(
        monthly > feature_stats["monthly_charge_75th"]
    )

    is_fiber = int(raw["InternetService"] == "Fiber optic")
    is_electronic_check = int(
        raw["PaymentMethod"] == "Electronic check"
    )

    tenure_contract_interaction = tenure * contract_strength

    high_charge_no_security = int(
        high_monthly_charge == 1
        and raw["OnlineSecurity"] == "No"
    )

    # ---- Build final feature dict ----
    full_features = raw.copy()

    full_features.update({
        "charges_per_month": charges_per_month,
        "tenure_bucket": tenure_bucket,
        "clv_proxy": clv_proxy,
        "avg_charge_per_tenure": avg_charge_per_tenure,
        "num_services": num_services,
        "service_density": service_density,
        "is_fiber": is_fiber,
        "contract_strength": contract_strength,
        "high_monthly_charge": high_monthly_charge,
        "is_electronic_check": is_electronic_check,
        "tenure_contract_interaction": tenure_contract_interaction,
        "high_charge_no_security": high_charge_no_security,
    })

    # ---- Ordered numpy vector ----
    vector = []

    for col in feature_columns:
        val = full_features[col]

        if col in categorical_columns:
            # ✅ SAFE lookup (prevents KeyError like 'gender')
            val = category_maps.get(col, {}).get(str(val), 0)

        vector.append(val)

    return np.array(vector, dtype=np.float32).reshape(1, -1)