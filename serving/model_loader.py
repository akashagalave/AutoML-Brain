import json
import os
from pathlib import Path
import lightgbm as lgb

_booster = None
_threshold = None
_feature_columns = None
_feature_stats = None
_categorical_columns = None
_categorical_indices = None
_category_maps = None


def load_model_assets():
    global _booster, _threshold, _feature_columns, _feature_stats, _categorical_columns, _categorical_indices, _category_maps

    if _booster is not None:
        return (
            _booster,
            _threshold,
            _feature_columns,
            _feature_stats,
            _categorical_columns,
            _categorical_indices,
            _category_maps,
        )

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "models" / "model_artifacts"

    _booster = lgb.Booster(
        model_file=str(model_dir / "model.txt")
    )

    with open(model_dir / "best_threshold.json") as f:
        _threshold = json.load(f)["threshold"]

    with open(model_dir / "feature_columns.json") as f:
        _feature_columns = json.load(f)

    with open(model_dir / "categorical_columns.json") as f:
        _categorical_columns = json.load(f)

    with open(model_dir / "feature_stats.json") as f:
        _feature_stats = json.load(f)

    # Build categorical index lookup
    _categorical_indices = [
        _feature_columns.index(col)
        for col in _categorical_columns
    ]

    # Extract category maps from trained model
    model_info = _booster.dump_model()
    _category_maps = {}

    for feat in model_info["feature_infos"]:
        if "categorical_values" in model_info["feature_infos"][feat]:
            cats = model_info["feature_infos"][feat]["categorical_values"]
            _category_maps[feat] = {str(v): i for i, v in enumerate(cats)}

    return (
        _booster,
        _threshold,
        _feature_columns,
        _feature_stats,
        _categorical_columns,
        _categorical_indices,
        _category_maps,
    )