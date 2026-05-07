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
    return 8 <= int(hour) <= 15


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

    if _has_enough_low_price_samples(y_train, cv_folds):
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

    return candidates


def _has_enough_low_price_samples(
    y_train: Sequence[float],
    cv_folds: Iterable[Tuple[np.ndarray, np.ndarray, str]] | None,
) -> bool:
    y_arr = np.asarray(y_train, dtype=float)
    low_mask = y_arr <= LOW_PRICE_THRESHOLD
    if int(low_mask.sum()) < MIN_LOW_PRICE_TOTAL:
        return False

    if cv_folds is None:
        return True

    for train_idx, _, _ in cv_folds:
        if int(low_mask[train_idx].sum()) < MIN_LOW_PRICE_PER_TRAIN_FOLD:
            return False
    return True


def _unique_groups(groups: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for group in groups:
        if group not in seen:
            seen.add(group)
            result.append(group)
    return result
