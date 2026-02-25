import os
import json
import lightgbm as lgb
import shap
import mlflow
from mlflow.tracking import MlflowClient
from tempfile import TemporaryDirectory

_booster = None
_threshold = None
_feature_columns = None
_feature_stats = None
_categorical_columns = None
_categorical_indices = None
_category_maps = None
_explainer = None


def load_model_assets():
    global _booster, _threshold, _feature_columns, _feature_stats
    global _categorical_columns, _categorical_indices
    global _category_maps, _explainer

    if _booster is not None:
        return (
            _booster,
            _threshold,
            _feature_columns,
            _feature_stats,
            _categorical_columns,
            _categorical_indices,
            _category_maps,
            _explainer,
        )

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    model_stage = os.getenv("MODEL_STAGE", "Production")
    model_uri = f"models:/churn_model/{model_stage}"

    print(f"Loading model from MLflow stage: {model_stage}")

   
    local_path = mlflow.artifacts.download_artifacts(model_uri)

    model_file = os.path.join(local_path, "model.txt")
    threshold_file = os.path.join(local_path, "best_threshold.json")
    feature_cols_file = os.path.join(local_path, "feature_columns.json")
    categorical_cols_file = os.path.join(local_path, "categorical_columns.json")
    feature_stats_file = os.path.join(local_path, "feature_stats.json")

    _booster = lgb.Booster(model_file=model_file)
    _explainer = shap.TreeExplainer(_booster)

    with open(threshold_file) as f:
        _threshold = json.load(f)["threshold"]

    with open(feature_cols_file) as f:
        _feature_columns = json.load(f)

    with open(categorical_cols_file) as f:
        _categorical_columns = json.load(f)

    with open(feature_stats_file) as f:
        _feature_stats = json.load(f)

    _categorical_indices = [
        _feature_columns.index(col)
        for col in _categorical_columns
    ]

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
        _explainer,
    )