import json
import os
from pathlib import Path
import lightgbm as lgb

_booster = None
_threshold = None
_feature_columns = None
_feature_stats = None


def load_model_assets():
    """
    Production-safe model loader.
    Ensures:
    - Model loaded once per worker
    - No thread oversubscription
    - Stable latency under concurrency
    """

    global _booster, _threshold, _feature_columns, _feature_stats

    if _booster is not None:
        return _booster, _threshold, _feature_columns, _feature_stats

    # ==============================
    # HARD THREAD CONTROL
    # ==============================

    # Prevent OpenMP oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "models" / "model_artifacts"

    # ==============================
    # LOAD LIGHTGBM BOOSTER
    # ==============================

    _booster = lgb.Booster(
        model_file=str(model_dir / "model.txt")
    )

    # Force single-thread inference inside LightGBM
    _booster.params["num_threads"] = 1

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

    # ==============================
    # LOAD FEATURE STATISTICS
    # ==============================

    with open(model_dir / "feature_stats.json") as f:
        _feature_stats = json.load(f)

    return _booster, _threshold, _feature_columns, _feature_stats