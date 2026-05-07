"""
Targeted LightGBM probe optimizer for monthly forward scenarios.

The optimizer is intentionally separate from normal training. It searches
hour-specific LightGBM probe parameters using only data available before the
target month. Candidate scoring is done on the last month inside the training
window, not on the target test month, and the optimizer does not rewrite default
model parameters automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config
from data_split import split_by_months
from feature_engineering_direct import DirectFeatureEngineer
from feature_selector import FeatureSelector
from model_factory import create_model, get_default_params, list_model_types
from utils.metrics import calculate_accuracy_rate, calculate_mae, calculate_rmse, calculate_smape


logger = logging.getLogger(__name__)


DEFAULT_FEATURE_GROUPS = ["direct_time", "direct_price_lag", "direct_market_window"]
WEATHER_FEATURE_GROUPS = [
    "direct_time",
    "direct_price_lag",
    "direct_market_window",
    "direct_weather_window",
]
MIDDAY_TIME_FEATURE_GROUP = "direct_time_midday"
MIDDAY_FEATURE_GROUPS = [
    MIDDAY_TIME_FEATURE_GROUP,
    "direct_price_lag",
    "direct_market_window",
    "direct_midday_regime",
]
MIDDAY_WEATHER_AGG_FEATURE_GROUPS = [
    MIDDAY_TIME_FEATURE_GROUP,
    "direct_price_lag",
    "direct_market_window",
    "direct_midday_regime",
    "direct_midday_weather_agg",
]
CALENDAR_FEATURE_GROUPS = [
    MIDDAY_TIME_FEATURE_GROUP,
    "direct_price_lag",
    "direct_market_window",
]
CALENDAR_WEATHER_FEATURE_GROUPS = [
    MIDDAY_TIME_FEATURE_GROUP,
    "direct_price_lag",
    "direct_market_window",
    "direct_weather_window",
]
MIDDAY_HOURS = set(range(8, 16))
NON_MIDDAY_EXPERIMENT_HOURS = set(range(0, 8)) | set(range(16, 24))
NON_MIDDAY_TWO_STAGE_HOURS = {16, 22, 23}
WEAK_HOURS = set(range(8, 17)) | {22, 23}

OBJECTIVES = ["regression", "regression_l1", "huber", "quantile"]
TREE_VALUES: Dict[str, List[Any]] = {
    "n_estimators": [120, 200, 300, 500],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "max_depth": [4, 5, 6, 8],
    "min_child_samples": [5, 10, 20, 30],
}
TREE_PROFILES: List[Dict[str, Any]] = [
    {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_child_samples": 20},
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10},
    {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "max_depth": 5, "min_child_samples": 10},
    {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8, "min_child_samples": 10},
]


class LightGBMProbeOptimizer:
    """Search probe parameters independently for each target hour."""

    def __init__(
        self,
        config: Config,
        model_type: str,
        test_months: Optional[List[str]],
        max_candidates: int,
        cv_folds: int,
        local_alpha_radius: float,
        local_alpha_step: float,
        broad_alpha_step: float,
    ):
        if model_type not in list_model_types() or not model_type.startswith("lightgbm"):
            raise ValueError(f"optimize-probe only supports LightGBM model types, got: {model_type}")

        self.config = config
        self.model_type = model_type
        self.test_months = test_months
        self.max_candidates = max_candidates
        self.cv_folds = max(1, cv_folds)
        self.local_alpha_radius = local_alpha_radius
        self.local_alpha_step = local_alpha_step
        self.broad_alpha_step = broad_alpha_step
        self.engineer = DirectFeatureEngineer()
        self.feature_selector = FeatureSelector()

    def optimize(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        if hours is None:
            hours = list(range(24))

        rows: List[Dict[str, Any]] = []
        for hour in hours:
            logger.info("Optimize probe H%02d", hour)
            try:
                rows.extend(self._optimize_hour(hour))
            except Exception as exc:
                logger.exception("Probe optimization failed: H%02d", hour)
                rows.append({"hour": hour, "status": "failed", "error": str(exc)})

        results_df = pd.DataFrame(rows)
        self._save_outputs(results_df, hours)
        return results_df

    def _optimize_hour(self, hour: int) -> List[Dict[str, Any]]:
        data_by_variant = {
            "default": self._prepare_data_by_month(hour, DEFAULT_FEATURE_GROUPS),
        }
        if hour in WEAK_HOURS or hour in NON_MIDDAY_EXPERIMENT_HOURS:
            data_by_variant["weather"] = self._prepare_data_by_month(hour, WEATHER_FEATURE_GROUPS)
        if hour in NON_MIDDAY_EXPERIMENT_HOURS:
            data_by_variant["calendar"] = self._prepare_data_by_month(hour, CALENDAR_FEATURE_GROUPS)
            data_by_variant["calendar_weather"] = self._prepare_data_by_month(hour, CALENDAR_WEATHER_FEATURE_GROUPS)
        if hour in MIDDAY_HOURS:
            data_by_variant["midday_regime"] = self._prepare_data_by_month(hour, MIDDAY_FEATURE_GROUPS)
            data_by_variant["midday_regime_weather"] = self._prepare_data_by_month(hour, MIDDAY_WEATHER_AGG_FEATURE_GROUPS)
            data_by_variant["midday_regime_weighted"] = self._prepare_data_by_month(hour, MIDDAY_FEATURE_GROUPS)

        base_params = get_default_params(self.model_type, hour=hour)
        if hour in MIDDAY_HOURS:
            candidates = self._build_candidates(base_params, hour in WEAK_HOURS)
            candidates = self._build_midday_candidates(candidates, base_params)
        elif hour in NON_MIDDAY_EXPERIMENT_HOURS:
            candidates = self._build_non_midday_candidates(base_params, hour)
        else:
            candidates = self._build_candidates(base_params, hour in WEAK_HOURS)
        rows = []

        for feature_variant, data_by_month in data_by_variant.items():
            for idx, params in enumerate(candidates, start=1):
                params = params.copy()
                if feature_variant == "midday_regime_weighted" and "sample_weight_mode" not in params and params.get("model_kind") != "two_stage_low_price":
                    params["sample_weight_mode"] = "default"
                start = time.time()
                row = {
                    "hour": hour,
                    "status": "success",
                    "candidate_id": idx,
                    "feature_variant": feature_variant,
                    "feature_count": len(next(iter(data_by_month.values()))["feature_cols"]),
                    "test_period": ",".join(data_by_month.keys()),
                    "target_period": ",".join(data_by_month.keys()),
                    "validation_period": ",".join(
                        sorted({month for data in data_by_month.values() for month in data["validation_months"]})
                    ),
                    "n_train": int(np.mean([fold["n_fit"] for data in data_by_month.values() for fold in data["folds"]])),
                    "n_validation": int(np.sum([data["split_info"]["n_validation"] for data in data_by_month.values()])),
                    "n_target_test": int(np.sum([data["split_info"]["n_target_test"] for data in data_by_month.values()])),
                    "n_cv_folds": int(np.sum([len(data["folds"]) for data in data_by_month.values()])),
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "is_default_params": idx == 1,
                }
                try:
                    monthly_metrics = self._evaluate_candidate_by_month(params, data_by_month)
                    smapes = [item["smape"] for item in monthly_metrics.values()]
                    acc_rates = [item["acc_rate"] for item in monthly_metrics.values()]
                    row.update(
                        {
                            "mae": float(np.mean([item["mae"] for item in monthly_metrics.values()])),
                            "rmse": float(np.mean([item["rmse"] for item in monthly_metrics.values()])),
                            "smape": float(np.mean(smapes)),
                            "max_month_smape": float(np.max(smapes)),
                            "min_month_smape": float(np.min(smapes)),
                            "smape_std": float(np.std(smapes)),
                            "acc_rate": float(np.mean(acc_rates)),
                            "generalization_score": self._generalization_score(smapes, acc_rates),
                            "monthly_metrics": json.dumps(monthly_metrics, ensure_ascii=False, sort_keys=True),
                            "training_time": time.time() - start,
                        }
                    )
                except Exception as exc:
                    row.update({"status": "failed", "error": str(exc), "training_time": time.time() - start})
                rows.append(row)

        return rows

    def _prepare_data_by_month(self, hour: int, feature_groups: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        features_df, target_col = self.engineer.load_features(hour)
        dates = pd.to_datetime(features_df["预测日期"])
        available_months = sorted(str(month) for month in dates.dt.to_period("M").unique())
        if self.test_months:
            requested_months = [str(month) for month in self.test_months]
        else:
            requested_months = [available_months[-1]]

        return {month: self._prepare_data_from_frame(features_df, target_col, hour, feature_groups, month) for month in requested_months}

    def _prepare_data_from_frame(
        self,
        features_df: pd.DataFrame,
        target_col: str,
        hour: int,
        feature_groups: Sequence[str],
        test_month: str,
    ) -> Dict[str, Any]:
        candidate_features = [col for col in features_df.columns if col not in [target_col, "预测日期"]]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = self.feature_selector.select_features_from_groups(list(feature_groups), numeric_features)
        split = split_by_months(features_df, "预测日期", [test_month])
        dates = pd.to_datetime(features_df["预测日期"])
        months = dates.dt.to_period("M")
        train_months = months[split.train_mask]
        validation_months = sorted(train_months.unique())[1:][-self.cv_folds :]
        if not validation_months:
            raise ValueError(f"目标月份 {test_month} 的训练窗口内没有可用的时间序列交叉验证月份")

        folds = []
        for validation_month in validation_months:
            fit_mask = split.train_mask & (months < validation_month)
            validation_mask = split.train_mask & (months == validation_month)
            if not fit_mask.any() or not validation_mask.any():
                continue
            folds.append(
                {
                    "validation_month": str(validation_month),
                    "X_fit": features_df.loc[fit_mask, feature_cols].reset_index(drop=True),
                    "y_fit": features_df.loc[fit_mask, target_col].to_numpy(),
                    "X_validation": features_df.loc[validation_mask, feature_cols].reset_index(drop=True),
                    "y_validation": features_df.loc[validation_mask, target_col].to_numpy(),
                    "fit_start": dates[fit_mask].min().strftime("%Y-%m-%d"),
                    "fit_end": dates[fit_mask].max().strftime("%Y-%m-%d"),
                    "validation_start": dates[validation_mask].min().strftime("%Y-%m-%d"),
                    "validation_end": dates[validation_mask].max().strftime("%Y-%m-%d"),
                    "n_fit": int(fit_mask.sum()),
                    "n_validation": int(validation_mask.sum()),
                }
            )

        if not folds:
            raise ValueError(f"目标月份 {test_month} 的训练窗口内没有有效的时间序列交叉验证 fold")

        return {
            "folds": folds,
            "feature_cols": feature_cols,
            "target_month": str(test_month),
            "validation_months": [fold["validation_month"] for fold in folds],
            "split_info": {
                **split.to_dict(),
                "target_month": str(test_month),
                "validation_months": [fold["validation_month"] for fold in folds],
                "cv_folds": len(folds),
                "folds": [
                    {
                        key: fold[key]
                        for key in [
                            "validation_month",
                            "fit_start",
                            "fit_end",
                            "validation_start",
                            "validation_end",
                            "n_fit",
                            "n_validation",
                        ]
                    }
                    for fold in folds
                ],
                "n_fit": int(np.mean([fold["n_fit"] for fold in folds])),
                "n_validation": int(np.sum([fold["n_validation"] for fold in folds])),
                "n_target_test": int(split.test_mask.sum()),
            },
        }

    def _evaluate_candidate_by_month(self, params: Dict[str, Any], data_by_month: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        monthly_metrics: Dict[str, Dict[str, float]] = {}
        for month, data in data_by_month.items():
            for fold in data["folds"]:
                model = create_model(self.model_type, params)
                model.fit(fold["X_fit"], fold["y_fit"])
                pred = model.predict(fold["X_validation"])
                key = f"{month}|valid={fold['validation_month']}"
                monthly_metrics[key] = {
                    "target_month": data["target_month"],
                    "validation_month": fold["validation_month"],
                    "mae": calculate_mae(fold["y_validation"], pred),
                    "rmse": calculate_rmse(fold["y_validation"], pred),
                    "smape": calculate_smape(fold["y_validation"], pred),
                    "acc_rate": calculate_accuracy_rate(fold["y_validation"], pred, threshold=20.0),
                    "n_fit": float(fold["n_fit"]),
                    "n_validation": float(fold["n_validation"]),
                    "n_target_test": float(data["split_info"]["n_target_test"]),
                }
        return monthly_metrics

    @staticmethod
    def _generalization_score(smapes: Sequence[float], acc_rates: Sequence[float]) -> float:
        smapes_arr = np.asarray(smapes, dtype=float)
        acc_arr = np.asarray(acc_rates, dtype=float)
        instability_penalty = max(0.0, float(np.max(smapes_arr) - np.mean(smapes_arr))) * 0.25
        acc_bonus = float(np.mean(acc_arr)) * 0.02
        return float(np.mean(smapes_arr) + instability_penalty - acc_bonus)

    def _build_candidates(self, base_params: Dict[str, Any], broad_search: bool) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = [self._normalize_params(base_params)]
        current_alpha = float(base_params.get("alpha", 0.5))

        for objective in OBJECTIVES:
            params = {**base_params, "objective": objective}
            if objective == "quantile":
                params["alpha"] = current_alpha
            candidates.append(self._normalize_params(params))

        for alpha in self._alpha_grid(current_alpha, self.local_alpha_radius, self.local_alpha_step):
            params = {**base_params, "objective": "quantile", "alpha": alpha}
            candidates.append(self._normalize_params(params))

        for key, values in TREE_VALUES.items():
            for value in values:
                params = {**base_params, key: value}
                candidates.append(self._normalize_params(params))

        if broad_search:
            for alpha in self._alpha_grid(0.5, 0.45, self.broad_alpha_step):
                params = {**base_params, "objective": "quantile", "alpha": alpha}
                candidates.append(self._normalize_params(params))

            for profile in TREE_PROFILES:
                for objective in ["regression", "regression_l1", "huber"]:
                    params = {**base_params, **profile, "objective": objective}
                    candidates.append(self._normalize_params(params))
                for alpha in self._alpha_grid(current_alpha, self.local_alpha_radius, self.local_alpha_step):
                    params = {**base_params, **profile, "objective": "quantile", "alpha": alpha}
                    candidates.append(self._normalize_params(params))

        return self._dedupe_candidates(candidates)[: self.max_candidates]

    def _build_midday_candidates(self, candidates: List[Dict[str, Any]], base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = list(candidates)
        seed_params = self._dedupe_candidates([self._normalize_params(base_params), *candidates[:30]])

        for params in seed_params:
            for mode in ["light", "default", "strong"]:
                weighted = params.copy()
                weighted["sample_weight_mode"] = mode
                result.append(weighted)

        two_stage_profiles = [
            self._normalize_params(base_params),
            {"objective": "regression_l1", "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
            {"objective": "regression_l1", "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
            {"objective": "quantile", "alpha": 0.2, "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
            {"objective": "quantile", "alpha": 0.8, "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
        ]
        for profile in self._dedupe_candidates(two_stage_profiles):
            for threshold in [50, 80, 120]:
                for prob_threshold in [0.35, 0.5, 0.65]:
                    for blend in [0.4, 0.7, 1.0]:
                        params = profile.copy()
                        params.update(
                            {
                                "model_kind": "two_stage_low_price",
                                "low_price_threshold": threshold,
                                "prob_threshold": prob_threshold,
                                "blend": blend,
                                "sample_weight_mode": "default",
                            }
                        )
                        result.append(params)

        return self._dedupe_candidates(result)[: self.max_candidates]

    def _build_non_midday_candidates(self, base_params: Dict[str, Any], hour: int) -> List[Dict[str, Any]]:
        """Compact search space for already-strong non-midday hours."""
        result: List[Dict[str, Any]] = []
        base = self._normalize_params(base_params)
        current_alpha = float(base_params.get("alpha", 0.5))

        result.append(base)
        for objective in OBJECTIVES:
            params = {**base_params, "objective": objective}
            if objective == "quantile":
                params["alpha"] = current_alpha
            result.append(self._normalize_params(params))

        for alpha in self._alpha_grid(current_alpha, min(self.local_alpha_radius, 0.08), max(self.local_alpha_step, 0.02)):
            result.append(self._normalize_params({**base_params, "objective": "quantile", "alpha": alpha}))

        for profile in TREE_PROFILES[:3]:
            result.append(self._normalize_params({**base_params, **profile}))
            result.append(self._normalize_params({**base_params, **profile, "objective": "regression"}))
            result.append(self._normalize_params({**base_params, **profile, "objective": "regression_l1"}))

        weighted_seeds = self._dedupe_candidates(result[:8])
        for params in weighted_seeds:
            for mode in ["light", "default", "strong"]:
                weighted = params.copy()
                weighted["sample_weight_mode"] = mode
                result.append(weighted)

        if hour in NON_MIDDAY_TWO_STAGE_HOURS:
            two_stage_profiles = self._dedupe_candidates(
                [
                    base,
                    self._normalize_params(
                        {
                            **base_params,
                            "objective": "regression_l1",
                            "n_estimators": 200,
                            "learning_rate": 0.05,
                            "num_leaves": 31,
                            "max_depth": 6,
                            "min_child_samples": 20,
                        }
                    ),
                    self._normalize_params(
                        {
                            **base_params,
                            "objective": "quantile",
                            "alpha": min(0.9, max(0.1, current_alpha)),
                            "n_estimators": 300,
                            "learning_rate": 0.03,
                            "num_leaves": 31,
                            "max_depth": 6,
                            "min_child_samples": 10,
                        }
                    ),
                ]
            )
            for profile in two_stage_profiles:
                for threshold in [80, 120]:
                    for prob_threshold in [0.35, 0.5]:
                        for blend in [0.4, 0.7]:
                            params = profile.copy()
                            params.update(
                                {
                                    "model_kind": "two_stage_low_price",
                                    "low_price_threshold": threshold,
                                    "prob_threshold": prob_threshold,
                                    "blend": blend,
                                    "sample_weight_mode": "default",
                                }
                            )
                            result.append(params)

        return self._dedupe_candidates(result)[: self.max_candidates]

    @staticmethod
    def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = params.copy()
        if normalized.get("objective") != "quantile":
            normalized.pop("alpha", None)
        normalized.setdefault("subsample", 0.8)
        normalized.setdefault("colsample_bytree", 0.8)
        normalized.setdefault("random_state", 42)
        return normalized

    @staticmethod
    def _dedupe_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for params in candidates:
            key = json.dumps(params, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            result.append(params)
        return result

    @staticmethod
    def _alpha_grid(center: float, radius: float, step: float) -> List[float]:
        low = max(0.05, center - radius)
        high = min(0.95, center + radius)
        values = np.arange(low, high + step / 2, step)
        return [float(round(value, 2)) for value in values]

    def _save_outputs(self, results_df: pd.DataFrame, requested_hours: List[int]) -> None:
        success_df = results_df[results_df["status"] == "success"].copy()
        if success_df.empty:
            raise RuntimeError("No successful optimization rows to save")

        test_period = str(success_df["test_period"].iloc[0])
        validation_period = str(success_df["validation_period"].iloc[0]) if "validation_period" in success_df.columns else ""
        safe_period = test_period.replace(",", "_")
        hour_suffix = ""
        if sorted(requested_hours) != list(range(24)):
            hour_suffix = "_" + "-".join(f"H{hour:02d}" for hour in sorted(requested_hours))
        log_dir = self.config.get_result_path("logs") / "direct" / self.model_type
        log_dir.mkdir(parents=True, exist_ok=True)

        csv_path = log_dir / f"probe_optimization_{safe_period}{hour_suffix}.csv"
        results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        score_col = "generalization_score" if "generalization_score" in success_df.columns else "smape"
        best_df = success_df.sort_values(["hour", score_col, "smape"]).groupby("hour", as_index=False).head(1)
        default_df = success_df[success_df["is_default_params"] & (success_df["feature_variant"] == "default")]

        summary = self._build_summary(best_df, default_df, requested_hours, test_period, validation_period)
        summary_path = log_dir / f"probe_optimization_summary{hour_suffix}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        best_df.to_csv(log_dir / f"probe_optimization_best_{safe_period}{hour_suffix}.csv", index=False, encoding="utf-8-sig")
        if not hour_suffix:
            with open(log_dir / "probe_optimization_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Probe optimization report saved: %s", csv_path)

    def _build_summary(
        self,
        best_df: pd.DataFrame,
        default_df: pd.DataFrame,
        requested_hours: List[int],
        test_period: str,
        validation_period: str,
    ) -> Dict[str, Any]:
        baseline_smape = float(default_df["smape"].mean())
        best_smape = float(best_df["smape"].mean())
        baseline_under20 = int((default_df["smape"] < 20).sum())
        best_under20 = int((best_df["smape"] < 20).sum())
        baseline_acc_rate = float(default_df["acc_rate"].mean()) if "acc_rate" in default_df.columns else None
        best_acc_rate = float(best_df["acc_rate"].mean()) if "acc_rate" in best_df.columns else None

        requested_set = set(requested_hours)
        midday_hours = sorted(requested_set & set(range(8, 17)))
        if midday_hours:
            baseline_midday = float(default_df[default_df["hour"].isin(midday_hours)]["smape"].mean())
            best_midday = float(best_df[best_df["hour"].isin(midday_hours)]["smape"].mean())
        else:
            baseline_midday = None
            best_midday = None

        no_month_regression = self._no_month_regression(best_df, default_df, tolerance=2.0)
        acceptance_reasons = []
        if best_smape < baseline_smape and no_month_regression:
            acceptance_reasons.append("multi_month_smape_improved_without_month_regression_gt_2")
        if best_acc_rate is not None and baseline_acc_rate is not None and best_acc_rate > baseline_acc_rate and no_month_regression:
            acceptance_reasons.append("multi_month_acc_rate_improved_without_month_regression_gt_2")
        if (
            baseline_midday is not None
            and best_midday is not None
            and baseline_midday - best_midday >= 3.0
            and no_month_regression
        ):
            acceptance_reasons.append("midday_smape_down_3_without_month_regression_gt_2")

        best_params = {}
        for row in best_df.sort_values("hour").itertuples(index=False):
            best_params[f"H{int(row.hour):02d}"] = {
                "smape": float(row.smape),
                "generalization_score": float(getattr(row, "generalization_score", row.smape)),
                "acc_rate": float(getattr(row, "acc_rate", np.nan)),
                "feature_variant": row.feature_variant,
                "monthly_metrics": json.loads(row.monthly_metrics) if hasattr(row, "monthly_metrics") else {},
                "params": json.loads(row.params),
            }

        return {
            "model_type": self.model_type,
            "test_period": test_period,
            "target_period": test_period,
            "validation_period": validation_period,
            "hours": requested_hours,
            "baseline_smape": baseline_smape,
            "best_smape": best_smape,
            "baseline_acc_rate": baseline_acc_rate,
            "best_acc_rate": best_acc_rate,
            "baseline_under20_hours": baseline_under20,
            "best_under20_hours": best_under20,
            "baseline_midday_smape": baseline_midday,
            "best_midday_smape": best_midday,
            "no_month_regression_gt_2": no_month_regression,
            "accepted": bool(acceptance_reasons),
            "acceptance_reasons": acceptance_reasons,
            "best_params_by_hour": best_params,
        }

    @staticmethod
    def _no_month_regression(best_df: pd.DataFrame, default_df: pd.DataFrame, tolerance: float) -> bool:
        default_by_hour = {int(row.hour): json.loads(row.monthly_metrics) for row in default_df.itertuples(index=False)}
        for row in best_df.itertuples(index=False):
            hour = int(row.hour)
            baseline_metrics = default_by_hour.get(hour, {})
            best_metrics = json.loads(row.monthly_metrics)
            for month, metrics in best_metrics.items():
                baseline_smape = baseline_metrics.get(month, {}).get("smape")
                if baseline_smape is None:
                    continue
                if float(metrics["smape"]) > float(baseline_smape) + tolerance:
                    return False
        return True


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "probe_optimizer.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize LightGBM sMAPE probe parameters on pre-target training validation")
    parser.add_argument("--model", default="lightgbm_smape_probe_v3", choices=list_model_types())
    parser.add_argument("--hours", type=int, nargs="+", default=None)
    parser.add_argument("--test-months", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--local-alpha-radius", type=float, default=0.10)
    parser.add_argument("--local-alpha-step", type=float, default=0.02)
    parser.add_argument("--broad-alpha-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    results = LightGBMProbeOptimizer(
        config=Config(),
        model_type=args.model,
        test_months=args.test_months,
        max_candidates=args.max_candidates,
        cv_folds=args.cv_folds,
        local_alpha_radius=args.local_alpha_radius,
        local_alpha_step=args.local_alpha_step,
        broad_alpha_step=args.broad_alpha_step,
    ).optimize(args.hours)
    ok = results[results["status"] == "success"]
    score_col = "generalization_score" if "generalization_score" in ok.columns else "smape"
    best = ok.sort_values(["hour", score_col, "smape"]).groupby("hour", as_index=False).head(1)
    display_cols = [col for col in ["hour", "feature_variant", "smape", "max_month_smape", "acc_rate", "generalization_score", "mae", "rmse"] if col in best.columns]
    print(best[display_cols].to_string(index=False))
    print(f"\nBest-combined average sMAPE={best['smape'].mean():.2f}%")


if __name__ == "__main__":
    main()
