"""Lightweight model registry for Direct forecasting."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


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


DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "lightgbm": LIGHTGBM_DEFAULT_PARAMS,
    "lightgbm_auto": LIGHTGBM_DEFAULT_PARAMS,
    "lightgbm_smape_probe": LIGHTGBM_DEFAULT_PARAMS,
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


LIGHTGBM_PARAM_SPACE: Dict[str, List[Any]] = {
    "objective": ["regression", "regression_l1", "huber", "quantile"],
    "alpha": [0.05, 0.10, 0.25, 0.35, 0.50, 0.60, 0.75, 0.90],
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
}


PARAM_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "lightgbm": LIGHTGBM_PARAM_SPACE,
    "lightgbm_auto": LIGHTGBM_PARAM_SPACE,
    "lightgbm_smape_probe": {},
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
    return DEFAULT_PARAMS[model_type].copy()


def get_param_space(model_type: str) -> Dict[str, List[Any]]:
    _validate_model_type(model_type)
    return PARAM_SPACES.get(model_type, {})


def create_model(model_type: str, params: Dict[str, Any]):
    _validate_model_type(model_type)
    params = params.copy()
    prediction_floor = params.pop("prediction_floor", None)
    params = _strip_training_metadata(params)

    if model_type in {"lightgbm", "lightgbm_auto", "lightgbm_smape_probe"}:
        from lightgbm import LGBMRegressor

        if params.get("model_kind") == "feature_ensemble":
            model = FeatureGroupEnsembleRegressor(params)
            return _with_prediction_bounds(model, prediction_floor)

        if params.get("model_kind") == "two_stage_low_price":
            model = LowPriceTwoStageRegressor(params)
            return _with_prediction_bounds(model, prediction_floor)

        if params.get("model_kind") == "low_price_gate":
            model = LowPriceGateRegressor(params)
            return _with_prediction_bounds(model, prediction_floor)

        sample_weight_mode = params.pop("sample_weight_mode", None)
        if sample_weight_mode:
            model = WeightedLGBMRegressor(params, sample_weight_mode)
            return _with_prediction_bounds(model, prediction_floor)

        params.pop("model_kind", None)
        model = LGBMRegressor(verbose=-1, **params)
        return _with_prediction_bounds(model, prediction_floor)

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


def _strip_training_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    params.pop("feature_groups", None)
    params.pop("candidate_name", None)
    params.pop("selected_structure", None)
    params.pop("tune_alpha", None)
    params.pop("tune_params", None)
    return params


def _with_prediction_bounds(model: Any, prediction_floor: Any = None):
    if prediction_floor is None:
        return model
    return PredictionBoundsRegressor(model, prediction_floor=float(prediction_floor))


def _smape_proxy_weights(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    base = 300.0 / (np.abs(y) + 30.0)
    if mode == "light":
        upper = 3.0
    elif mode == "strong":
        upper = 8.0
    else:
        upper = 6.0
    return np.clip(base, 1.0, upper)


class WeightedLGBMRegressor:
    """LightGBM wrapper that computes sMAPE-proxy sample weights from y."""

    def __init__(self, params: Dict[str, Any], sample_weight_mode: str):
        from lightgbm import LGBMRegressor

        self.params = _strip_training_metadata(params.copy())
        self.sample_weight_mode = sample_weight_mode
        self.model = LGBMRegressor(verbose=-1, **self.params)

    def fit(self, X, y):
        self.model.fit(X, y, sample_weight=_smape_proxy_weights(y, self.sample_weight_mode))
        return self

    def predict(self, X):
        return self.model.predict(X)


def _create_lgbm_from_params(params: Dict[str, Any]):
    params = params.copy()
    prediction_floor = params.pop("prediction_floor", None)
    params = _strip_training_metadata(params)
    if params.get("model_kind") == "two_stage_low_price":
        return _with_prediction_bounds(LowPriceTwoStageRegressor(params), prediction_floor)

    if params.get("model_kind") == "low_price_gate":
        return _with_prediction_bounds(LowPriceGateRegressor(params), prediction_floor)

    sample_weight_mode = params.pop("sample_weight_mode", None)
    if sample_weight_mode:
        return _with_prediction_bounds(WeightedLGBMRegressor(params, sample_weight_mode), prediction_floor)

    params.pop("model_kind", None)
    from lightgbm import LGBMRegressor

    return _with_prediction_bounds(LGBMRegressor(verbose=-1, **params), prediction_floor)


class PredictionBoundsRegressor:
    """Post-process predictions with a configurable lower bound."""

    def __init__(self, model: Any, prediction_floor: float):
        self.model = model
        self.prediction_floor = prediction_floor

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        pred = np.asarray(self.model.predict(X), dtype=float)
        return np.maximum(pred, self.prediction_floor)


class FeatureGroupEnsembleRegressor:
    """Weighted ensemble where each member can use a different feature group."""

    def __init__(self, params: Dict[str, Any]):
        config = params.copy()
        config.pop("model_kind", None)
        config.pop("feature_groups", None)
        self.members = config.pop("members")
        self.fitted_members_ = []

    def fit(self, X, y):
        from ..feature_engineering.selector import FeatureSelector

        selector = FeatureSelector()
        available_features = list(X.columns)
        total_weight = sum(float(member.get("weight", 1.0)) for member in self.members)
        if total_weight <= 0:
            raise ValueError("feature_ensemble requires positive member weights")

        self.fitted_members_ = []
        for member in self.members:
            weight = float(member.get("weight", 1.0)) / total_weight
            feature_groups = member.get("feature_groups")
            if feature_groups:
                feature_cols = selector.select_features_from_groups(feature_groups, available_features)
            else:
                feature_cols = available_features
            model = _create_lgbm_from_params(member["params"])
            model.fit(X[feature_cols], y)
            self.fitted_members_.append((weight, feature_cols, model))
        return self

    def predict(self, X):
        if not self.fitted_members_:
            raise ValueError("FeatureGroupEnsembleRegressor is not fitted")
        pred = None
        for weight, feature_cols, model in self.fitted_members_:
            member_pred = np.asarray(model.predict(X[feature_cols]), dtype=float)
            pred = weight * member_pred if pred is None else pred + weight * member_pred
        return pred


class LowPriceTwoStageRegressor:
    """Blend a main regressor with a low-price specialist based on low-price probability."""

    def __init__(self, params: Dict[str, Any]):
        from lightgbm import LGBMClassifier, LGBMRegressor

        config = _strip_training_metadata(params.copy())
        config.pop("model_kind", None)
        self.low_price_threshold = float(config.pop("low_price_threshold", 80.0))
        self.prob_threshold = float(config.pop("prob_threshold", 0.5))
        self.blend = float(config.pop("blend", 0.7))
        self.sample_weight_mode = config.pop("sample_weight_mode", None)
        self.low_sample_weight_mode = config.pop("low_sample_weight_mode", "default")
        self.base_params = config

        classifier_params = {
            key: value
            for key, value in self.base_params.items()
            if key not in {"objective", "alpha"}
        }
        classifier_params.setdefault("n_estimators", 120)
        classifier_params.setdefault("learning_rate", 0.05)
        classifier_params.setdefault("num_leaves", 15)
        classifier_params.setdefault("max_depth", 4)
        classifier_params.setdefault("min_child_samples", 10)
        classifier_params.setdefault("random_state", 42)

        self.main_model = LGBMRegressor(verbose=-1, **self.base_params)
        low_params = self.base_params.copy()
        low_params["objective"] = "regression_l1"
        low_params.pop("alpha", None)
        self.low_model = LGBMRegressor(verbose=-1, **low_params)
        self.classifier = LGBMClassifier(verbose=-1, **classifier_params)

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        low_mask = y_arr <= self.low_price_threshold
        sample_weight = _smape_proxy_weights(y_arr, self.sample_weight_mode) if self.sample_weight_mode else None
        self.main_model.fit(X, y_arr, sample_weight=sample_weight)

        class_weight = _smape_proxy_weights(y_arr, "default")
        self.classifier.fit(X, low_mask.astype(int), sample_weight=class_weight)

        if int(low_mask.sum()) >= 4:
            low_weight = _smape_proxy_weights(y_arr[low_mask], self.low_sample_weight_mode)
            self.low_model.fit(
                X.loc[low_mask] if hasattr(X, "loc") else X[low_mask],
                y_arr[low_mask],
                sample_weight=low_weight,
            )
            self.has_low_model_ = True
        else:
            self.has_low_model_ = False
        return self

    def predict(self, X):
        main_pred = np.asarray(self.main_model.predict(X), dtype=float)
        if not self.has_low_model_:
            return main_pred
        low_prob = self.classifier.predict_proba(X)[:, 1]
        low_pred = np.asarray(self.low_model.predict(X), dtype=float)
        low_pred = np.maximum(low_pred, 0.0)
        use_low = low_prob >= self.prob_threshold
        blended = main_pred.copy()
        blended[use_low] = (1.0 - self.blend) * main_pred[use_low] + self.blend * low_pred[use_low]
        return blended


class LowPriceGateRegressor:
    """Cap predictions when a classifier detects a low-price regime."""

    def __init__(self, params: Dict[str, Any]):
        from lightgbm import LGBMClassifier, LGBMRegressor

        config = _strip_training_metadata(params.copy())
        config.pop("model_kind", None)
        self.low_price_threshold = float(config.pop("low_price_threshold", 150.0))
        self.gate_prob_threshold = float(config.pop("gate_prob_threshold", 0.05))
        self.gate_prediction_cap = float(config.pop("gate_prediction_cap", 40.0))
        self.sample_weight_mode = config.pop("sample_weight_mode", None)
        self.classifier_weight_mode = config.pop("classifier_weight_mode", "default")
        self.base_params = config

        classifier_params = {
            key: value
            for key, value in self.base_params.items()
            if key not in {"objective", "alpha"}
        }
        classifier_params.setdefault("n_estimators", 120)
        classifier_params.setdefault("learning_rate", 0.05)
        classifier_params.setdefault("num_leaves", 15)
        classifier_params.setdefault("max_depth", 4)
        classifier_params.setdefault("min_child_samples", 10)
        classifier_params.setdefault("random_state", 42)

        self.main_model = LGBMRegressor(verbose=-1, **self.base_params)
        self.model = self.main_model
        self.classifier = LGBMClassifier(verbose=-1, **classifier_params)

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        sample_weight = _smape_proxy_weights(y_arr, self.sample_weight_mode) if self.sample_weight_mode else None
        self.main_model.fit(X, y_arr, sample_weight=sample_weight)

        low_mask = y_arr <= self.low_price_threshold
        self.has_gate_ = len(np.unique(low_mask)) == 2
        if self.has_gate_:
            class_weight = _smape_proxy_weights(y_arr, self.classifier_weight_mode)
            self.classifier.fit(X, low_mask.astype(int), sample_weight=class_weight)
        return self

    def predict(self, X):
        main_pred = np.asarray(self.main_model.predict(X), dtype=float)
        if not getattr(self, "has_gate_", False):
            return main_pred
        low_prob = self.classifier.predict_proba(X)[:, 1]
        gated = low_prob >= self.gate_prob_threshold
        pred = main_pred.copy()
        pred[gated] = np.minimum(pred[gated], self.gate_prediction_cap)
        return pred
