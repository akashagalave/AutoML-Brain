import numpy as np

# ==============================
# STATIC CONSTANTS (Module Load)
# ==============================

SERVICE_COLS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

CONTRACT_MAP = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}


def _tenure_bucket_fast(t: int) -> str:
    if t <= 12:
        return "0-1y"
    if t <= 24:
        return "1-2y"
    if t <= 48:
        return "2-4y"
    return "4-6y"


def build_feature_vector(
    raw: dict,
    feature_stats: dict,
    feature_columns: list,
    categorical_columns: list,
    category_maps: dict,
):
    """
    Ultra-optimized inference feature builder.

    - No dict copy
    - No dict update
    - No dynamic object allocation except final vector
    - Minimal branching
    """

    monthly_75 = feature_stats["monthly_charge_75th"]

    # --- Base numeric values ---
    tenure = int(raw["tenure"])
    monthly = float(raw["MonthlyCharges"])
    total_charges = float(raw["TotalCharges"])

    # --- Derived features ---
    charges_per_month = total_charges / (tenure + 1)
    clv_proxy = monthly * tenure
    avg_charge_per_tenure = total_charges / (tenure + 1)

    tenure_bucket = _tenure_bucket_fast(tenure)

    # Service aggregation
    num_services = 0
    for col in SERVICE_COLS:
        if raw[col] == "Yes":
            num_services += 1

    service_density = num_services / 6.0

    contract_strength = CONTRACT_MAP[raw["Contract"]]

    high_monthly_charge = 1 if monthly > monthly_75 else 0
    is_fiber = 1 if raw["InternetService"] == "Fiber optic" else 0
    is_electronic_check = 1 if raw["PaymentMethod"] == "Electronic check" else 0

    tenure_contract_interaction = tenure * contract_strength

    high_charge_no_security = (
        1 if (high_monthly_charge and raw["OnlineSecurity"] == "No") else 0
    )

    # --- Build vector directly ---
    vector = np.empty(len(feature_columns), dtype=np.float32)

    # Convert categorical list to set once (fast membership)
    categorical_set = set(categorical_columns)

    for i, col in enumerate(feature_columns):

        if col == "charges_per_month":
            val = charges_per_month
        elif col == "tenure_bucket":
            val = tenure_bucket
        elif col == "clv_proxy":
            val = clv_proxy
        elif col == "avg_charge_per_tenure":
            val = avg_charge_per_tenure
        elif col == "num_services":
            val = num_services
        elif col == "service_density":
            val = service_density
        elif col == "is_fiber":
            val = is_fiber
        elif col == "contract_strength":
            val = contract_strength
        elif col == "high_monthly_charge":
            val = high_monthly_charge
        elif col == "is_electronic_check":
            val = is_electronic_check
        elif col == "tenure_contract_interaction":
            val = tenure_contract_interaction
        elif col == "high_charge_no_security":
            val = high_charge_no_security
        else:
            val = raw[col]

        # Categorical encoding
        if col in categorical_set:
            val = category_maps.get(col, {}).get(str(val), 0)

        vector[i] = val

    return vector.reshape(1, -1)