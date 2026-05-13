"""Automatic structure candidates for Direct LightGBM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_GROUPS = ["direct_time", "direct_price_lag", "direct_market_window"]
BASE_WEATHER_GROUPS = [*BASE_GROUPS, "direct_weather_window"]
MIDDAY_BASE_GROUPS = ["direct_time_midday", "direct_price_lag", "direct_market_window"]
MIDDAY_REGIME_GROUPS = [*MIDDAY_BASE_GROUPS, "direct_midday_regime"]
MIDDAY_WEATHER_GROUPS = [*MIDDAY_REGIME_GROUPS, "direct_weather_window", "direct_midday_weather_agg"]

LOW_PRICE_THRESHOLD = 50.0
MIDDAY_LOW_PRICE_THRESHOLDS = (80.0, 100.0)
MIN_LOW_PRICE_TOTAL = 20
MIN_LOW_PRICE_PER_TRAIN_FOLD = 5


@dataclass(frozen=True)
class AutoCandidate:
    name: str
    structure: str
    feature_groups: List[str]
    params: Dict[str, Any]
    complexity_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "structure": self.structure,
            "feature_groups": self.feature_groups,
            "params": self.params,
            "complexity_penalty": self.complexity_penalty,
        }


def is_midday_hour(hour: int) -> bool:
    return 9 <= int(hour) <= 15


def monthly_time_series_folds(
    dates: Sequence[Any],
    cv_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Build expanding-window folds where each validation fold is one month."""
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    months = date_series.dt.to_period("M")
    unique_months = sorted(months.unique())
    if len(unique_months) < 2:
        raise ValueError("时间序列 CV 至少需要两个训练月份")

    validation_months = unique_months[1:]
    if cv_folds > 0:
        validation_months = validation_months[-cv_folds:]

    folds: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for validation_month in validation_months:
        train_idx = np.flatnonzero((months < validation_month).to_numpy())
        val_idx = np.flatnonzero((months == validation_month).to_numpy())
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx, str(validation_month)))

    if not folds:
        raise ValueError("无法生成有效的时间序列 CV 折")
    return folds


def generate_auto_candidates(
    hour: int,
    default_params: Dict[str, Any],
    y_train: Sequence[float],
    cv_folds: Iterable[Tuple[np.ndarray, np.ndarray, str]] | None = None,
) -> List[AutoCandidate]:
    """Generate a bounded set of structure/feature-group candidates."""
    candidates: List[AutoCandidate] = []

    if is_midday_hour(hour):
        primary_groups = MIDDAY_BASE_GROUPS
        weather_groups = MIDDAY_WEATHER_GROUPS
        secondary_groups = BASE_WEATHER_GROUPS
    else:
        primary_groups = BASE_GROUPS
        weather_groups = BASE_WEATHER_GROUPS
        secondary_groups = BASE_GROUPS

    candidates.append(
        AutoCandidate(
            name="single_base",
            structure="single_lgbm",
            feature_groups=list(primary_groups),
            params={**default_params, "feature_groups": list(primary_groups)},
            complexity_penalty=0.0,
        )
    )
    candidates.append(
        AutoCandidate(
            name="single_weather",
            structure="single_lgbm",
            feature_groups=list(weather_groups),
            params={**default_params, "feature_groups": list(weather_groups)},
            complexity_penalty=0.03,
        )
    )

    candidates.append(
        AutoCandidate(
            name="weighted_weather",
            structure="weighted_lgbm",
            feature_groups=list(weather_groups),
            params={**default_params, "feature_groups": list(weather_groups), "sample_weight_mode": "light"},
            complexity_penalty=0.08,
        )
    )

    if int(hour) == 8:
        candidates.append(
            AutoCandidate(
                name="h08_low_price_gate",
                structure="low_price_gate",
                feature_groups=list(weather_groups),
                params={
                    **default_params,
                    "model_kind": "low_price_gate",
                    "feature_groups": list(weather_groups),
                    "low_price_threshold": 150.0,
                    "gate_prob_threshold": 0.05,
                    "gate_prediction_cap": 40.0,
                    "classifier_weight_mode": "default",
                },
                complexity_penalty=0.06,
            )
        )

    if int(hour) == 13:
        candidates.append(
            AutoCandidate(
                name="h13_quantile_base_weather_high",
                structure="single_lgbm",
                feature_groups=list(BASE_WEATHER_GROUPS),
                params={
                    **default_params,
                    "objective": "quantile",
                    "alpha": 0.75,
                    "feature_groups": list(BASE_WEATHER_GROUPS),
                    "tune_alpha": False,
                    "tune_params": False,
                },
                complexity_penalty=0.02,
            )
        )

    if int(hour) == 9:
        candidates.append(
            AutoCandidate(
                name="h09_low_price_gate",
                structure="low_price_gate",
                feature_groups=list(weather_groups),
                params={
                    **default_params,
                    "model_kind": "low_price_gate",
                    "feature_groups": list(weather_groups),
                    "low_price_threshold": 80.0,
                    "gate_prob_threshold": 0.05,
                    "gate_prediction_cap": 40.0,
                    "classifier_weight_mode": "default",
                    "tune_params": False,
                },
                complexity_penalty=0.05,
            )
        )

    if int(hour) == 12:
        candidates.append(
            AutoCandidate(
                name="h12_quantile_base_weather_high",
                structure="single_lgbm",
                feature_groups=list(BASE_WEATHER_GROUPS),
                params={
                    **default_params,
                    "objective": "quantile",
                    "alpha": 0.75,
                    "feature_groups": list(BASE_WEATHER_GROUPS),
                    "tune_alpha": False,
                    "tune_params": False,
                },
                complexity_penalty=0.02,
            )
        )

    if int(hour) == 14:
        candidates.append(
            AutoCandidate(
                name="h14_weighted_weather_light_fixed",
                structure="weighted_lgbm",
                feature_groups=list(weather_groups),
                params={
                    **default_params,
                    "feature_groups": list(weather_groups),
                    "sample_weight_mode": "light",
                    "tune_params": False,
                },
                complexity_penalty=0.05,
            )
        )

    if is_midday_hour(hour):
        candidates.append(
            AutoCandidate(
                name="single_weather_floor",
                structure="single_lgbm",
                feature_groups=list(weather_groups),
                params={**default_params, "feature_groups": list(weather_groups), "prediction_floor": 20.0},
                complexity_penalty=0.04,
            )
        )

        for name, alpha, penalty in (
            ("quantile_low_weather", 0.35, 0.05),
            ("quantile_mid_weather", 0.60, 0.06),
            ("quantile_high_weather", 0.90, 0.08),
        ):
            candidates.append(
                AutoCandidate(
                    name=name,
                    structure="single_lgbm",
                    feature_groups=list(weather_groups),
                    params={
                        **_midday_quantile_params(default_params, alpha),
                        "feature_groups": list(weather_groups),
                    },
                    complexity_penalty=penalty,
                )
            )

        candidates.append(
            AutoCandidate(
                name="quantile_floor_weather",
                structure="single_lgbm",
                feature_groups=list(weather_groups),
                params={
                    **_midday_quantile_params(default_params, 0.60),
                    "feature_groups": list(weather_groups),
                    "prediction_floor": 20.0,
                },
                complexity_penalty=0.07,
            )
        )

        candidates.append(
            AutoCandidate(
                name="weighted_weather_strong",
                structure="weighted_lgbm",
                feature_groups=list(weather_groups),
                params={**default_params, "feature_groups": list(weather_groups), "sample_weight_mode": "strong"},
                complexity_penalty=0.10,
            )
        )

        candidates.append(
            AutoCandidate(
                name="weighted_weather_floor",
                structure="weighted_lgbm",
                feature_groups=list(weather_groups),
                params={
                    **default_params,
                    "feature_groups": list(weather_groups),
                    "sample_weight_mode": "strong",
                    "prediction_floor": 20.0,
                },
                complexity_penalty=0.11,
            )
        )

    if _has_enough_low_price_samples(y_train, cv_folds, LOW_PRICE_THRESHOLD):
        candidates.append(
            AutoCandidate(
                name="two_stage_low_price",
                structure="two_stage_low_price",
                feature_groups=list(weather_groups),
                params={
                    **default_params,
                    "model_kind": "two_stage_low_price",
                    "feature_groups": list(weather_groups),
                    "low_price_threshold": LOW_PRICE_THRESHOLD,
                    "prob_threshold": 0.5,
                    "blend": 0.7,
                    "sample_weight_mode": "light",
                },
                complexity_penalty=0.20,
            )
        )

    if is_midday_hour(hour):
        for threshold, penalty in zip(MIDDAY_LOW_PRICE_THRESHOLDS, (0.22, 0.24)):
            if not _has_enough_low_price_samples(y_train, cv_folds, threshold):
                continue
            candidates.append(
                AutoCandidate(
                    name=f"two_stage_low_price_{int(threshold)}",
                    structure="two_stage_low_price",
                    feature_groups=list(weather_groups),
                    params={
                        **_midday_quantile_params(default_params, 0.60),
                        "model_kind": "two_stage_low_price",
                        "feature_groups": list(weather_groups),
                        "low_price_threshold": threshold,
                        "prob_threshold": 0.5,
                        "blend": 0.7,
                        "sample_weight_mode": "strong",
                        "low_sample_weight_mode": "strong",
                    },
                    complexity_penalty=penalty,
                )
            )

        if _has_enough_low_price_samples(y_train, cv_folds, 120.0):
            candidates.append(
                AutoCandidate(
                    name="two_stage_low_price_120_aggressive",
                    structure="two_stage_low_price",
                    feature_groups=list(weather_groups),
                    params={
                        **_midday_quantile_params(default_params, 0.75),
                        "model_kind": "two_stage_low_price",
                        "feature_groups": list(weather_groups),
                        "low_price_threshold": 120.0,
                        "prob_threshold": 0.35,
                        "blend": 1.0,
                        "sample_weight_mode": "strong",
                        "low_sample_weight_mode": "strong",
                        "prediction_floor": 20.0,
                    },
                    complexity_penalty=0.28,
                )
            )

    ensemble_params = {
        "model_kind": "feature_ensemble",
        "members": [
            {"weight": 0.65, "feature_groups": list(weather_groups), "params": {**default_params}},
            {"weight": 0.35, "feature_groups": list(secondary_groups), "params": {**default_params}},
        ],
    }
    candidates.append(
        AutoCandidate(
            name="feature_ensemble_light",
            structure="feature_ensemble",
            feature_groups=_unique_groups([*weather_groups, *secondary_groups]),
            params=ensemble_params,
            complexity_penalty=0.25,
        )
    )

    if is_midday_hour(hour):
        quantile_ensemble_params = {
            "model_kind": "feature_ensemble",
            "members": [
                {
                    "weight": 0.45,
                    "feature_groups": list(weather_groups),
                    "params": {**_midday_quantile_params(default_params, 0.35), "sample_weight_mode": "strong"},
                },
                {
                    "weight": 0.35,
                    "feature_groups": list(weather_groups),
                    "params": _midday_quantile_params(default_params, 0.60),
                },
                {
                    "weight": 0.20,
                    "feature_groups": list(secondary_groups),
                    "params": {**default_params},
                },
            ],
        }
        candidates.append(
            AutoCandidate(
                name="feature_ensemble_quantile_mix",
                structure="feature_ensemble",
                feature_groups=_unique_groups([*weather_groups, *secondary_groups]),
                params=quantile_ensemble_params,
                complexity_penalty=0.30,
            )
        )

    return candidates


def _has_enough_low_price_samples(
    y_train: Sequence[float],
    cv_folds: Iterable[Tuple[np.ndarray, np.ndarray, str]] | None,
    threshold: float = LOW_PRICE_THRESHOLD,
) -> bool:
    y_arr = np.asarray(y_train, dtype=float)
    low_mask = y_arr <= threshold
    if int(low_mask.sum()) < MIN_LOW_PRICE_TOTAL:
        return False

    if cv_folds is None:
        return True

    for train_idx, _, _ in cv_folds:
        if int(low_mask[train_idx].sum()) < MIN_LOW_PRICE_PER_TRAIN_FOLD:
            return False
    return True


def _midday_quantile_params(default_params: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    params = {
        **default_params,
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": 120,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    return params


def _unique_groups(groups: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for group in groups:
        if group not in seen:
            seen.add(group)
            result.append(group)
    return result
