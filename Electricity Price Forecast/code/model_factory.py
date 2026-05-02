"""
Lightweight model registry for Direct forecasting.
"""

from typing import Any, Dict, List


LIGHTGBM_DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "regression_l1",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
}


LIGHTGBM_SMAPE_PROBE_PARAMS: Dict[int, Dict[str, Any]] = {
    0: {"objective": "regression", "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
    1: {"objective": "quantile", "alpha": 0.18, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    2: {"objective": "quantile", "alpha": 0.45, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    3: {"objective": "regression_l1", "n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "max_depth": 5, "min_child_samples": 10},
    4: {"objective": "quantile", "alpha": 0.45, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    5: {"objective": "quantile", "alpha": 0.35, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    6: {"objective": "regression", "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    7: {"objective": "quantile", "alpha": 0.05, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    8: {"objective": "quantile", "alpha": 0.08, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    9: {"objective": "quantile", "alpha": 0.18, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    10: {"objective": "quantile", "alpha": 0.18, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    11: {"objective": "quantile", "alpha": 0.8, "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
    12: {"objective": "quantile", "alpha": 0.8, "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
    13: {"objective": "quantile", "alpha": 0.8, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    14: {"objective": "quantile", "alpha": 0.76, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    15: {"objective": "quantile", "alpha": 0.45, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    16: {"objective": "quantile", "alpha": 0.28, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    17: {"objective": "quantile", "alpha": 0.18, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    18: {"objective": "quantile", "alpha": 0.25, "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
    19: {"objective": "quantile", "alpha": 0.15, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    20: {"objective": "quantile", "alpha": 0.6, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    21: {"objective": "quantile", "alpha": 0.05, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    22: {"objective": "quantile", "alpha": 0.2, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    23: {"objective": "quantile", "alpha": 0.08, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
}


LIGHTGBM_SMAPE_PROBE_V2_PARAMS: Dict[int, Dict[str, Any]] = {
    hour: params.copy() for hour, params in LIGHTGBM_SMAPE_PROBE_PARAMS.items()
}

LIGHTGBM_SMAPE_PROBE_V2_PARAMS.update(
    {
        8: {"objective": "quantile", "alpha": 0.07, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
        5: {"objective": "quantile", "alpha": 0.35, "n_estimators": 200, "learning_rate": 0.08, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
        7: {"objective": "quantile", "alpha": 0.05, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
        11: {"objective": "quantile", "alpha": 0.8, "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 30},
        12: {"objective": "quantile", "alpha": 0.72, "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
        13: {"objective": "quantile", "alpha": 0.8, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 30},
        14: {"objective": "quantile", "alpha": 0.76, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 6, "min_child_samples": 10},
        15: {"objective": "regression", "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
        16: {"objective": "regression", "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
        22: {"objective": "quantile", "alpha": 0.2, "n_estimators": 120, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 6, "min_child_samples": 10},
        23: {"objective": "quantile", "alpha": 0.08, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 30},
    }
)


def _with_common_lgbm_params(hourly_params: Dict[int, Dict[str, Any]]) -> None:
    for params in hourly_params.values():
        params.update({"subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42})


_with_common_lgbm_params(LIGHTGBM_SMAPE_PROBE_PARAMS)
_with_common_lgbm_params(LIGHTGBM_SMAPE_PROBE_V2_PARAMS)


DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "lightgbm": LIGHTGBM_DEFAULT_PARAMS,
    "lightgbm_smape_probe": LIGHTGBM_DEFAULT_PARAMS,
    "lightgbm_smape_probe_v2": LIGHTGBM_DEFAULT_PARAMS,
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "ridge": {
        "alpha": 1.0,
    },
    "lasso": {
        "alpha": 0.01,
        "max_iter": 10000,
    },
}


PARAM_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "lightgbm": {
        "objective": ["regression", "regression_l1", "huber"],
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0],
        "min_child_samples": [5, 10, 20, 30],
        "random_state": [42],
    },
    "lightgbm_smape_probe": {},
    "lightgbm_smape_probe_v2": {},
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "random_state": [42],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [6, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "random_state": [42],
        "n_jobs": [-1],
    },
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "max_iter": [10000],
    },
}


def list_model_types() -> List[str]:
    return sorted(DEFAULT_PARAMS.keys())


def get_default_params(model_type: str, hour: int | None = None) -> Dict[str, Any]:
    _validate_model_type(model_type)
    if model_type == "lightgbm_smape_probe" and hour in LIGHTGBM_SMAPE_PROBE_PARAMS:
        return LIGHTGBM_SMAPE_PROBE_PARAMS[hour].copy()
    if model_type == "lightgbm_smape_probe_v2" and hour in LIGHTGBM_SMAPE_PROBE_V2_PARAMS:
        return LIGHTGBM_SMAPE_PROBE_V2_PARAMS[hour].copy()
    return DEFAULT_PARAMS[model_type].copy()


def get_param_space(model_type: str) -> Dict[str, List[Any]]:
    _validate_model_type(model_type)
    return PARAM_SPACES.get(model_type, {})


def create_model(model_type: str, params: Dict[str, Any]):
    _validate_model_type(model_type)
    params = params.copy()

    if model_type in {"lightgbm", "lightgbm_smape_probe", "lightgbm_smape_probe_v2"}:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(verbose=-1, **params)

    if model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)

    if model_type == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(**params)

    raise ValueError(f"不支持的模型类型: {model_type}")


def _validate_model_type(model_type: str) -> None:
    if model_type not in DEFAULT_PARAMS:
        supported = ", ".join(list_model_types())
        raise ValueError(f"不支持的模型类型: {model_type}. 可选: {supported}")
