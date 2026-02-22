import json
import os
from pathlib import Path
import lightgbm as lgb

_booster = None
_threshold = None
_feature_columns = None
_feature_stats = None
_categorical_columns = None


def load_model_assets():
    """
    Production-safe model loader.

    Guarantees:
    - Single model load per worker
    - No thread oversubscription
    - Deterministic inference latency
    """

    global _booster, _threshold, _feature_columns, _feature_stats, _categorical_columns

    if _booster is not None:
        return (
            _booster,
            _threshold,
            _feature_columns,
            _feature_stats,
            _categorical_columns,
        )

    # ==============================
    # HARD THREAD CONTROL
    # ==============================
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "models" / "model_artifacts"

    # ==============================
    # LOAD LIGHTGBM MODEL
    # ==============================
    _booster = lgb.Booster(
        model_file=str(model_dir / "model.txt"),
        params={"num_threads": 1}
    )

    # ==============================
    # LOAD THRESHOLD
    # ==============================
    with open(model_dir / "best_threshold.json") as f:
        _threshold = json.load(f)["threshold"]

    # ==============================
    # LOAD FEATURE SCHEMA
    # ==============================
    with open(model_dir / "feature_columns.json") as f:
        _feature_columns = json.load(f)

    with open(model_dir / "categorical_columns.json") as f:
        _categorical_columns = json.load(f)

    # ==============================
    # LOAD FEATURE STATS
    # ==============================
    with open(model_dir / "feature_stats.json") as f:
        _feature_stats = json.load(f)

    return (
        _booster,
        _threshold,
        _feature_columns,
        _feature_stats,
        _categorical_columns,
    )