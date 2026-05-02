"""
Targeted LightGBM probe optimizer for the latest monthly holdout.

The optimizer is intentionally separate from normal training. It searches
hour-specific LightGBM probe parameters on a chosen month and writes the search
trace, but it does not rewrite default model parameters automatically.
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
from utils.metrics import calculate_mae, calculate_rmse, calculate_smape


logger = logging.getLogger(__name__)


DEFAULT_FEATURE_GROUPS = ["direct_time", "direct_price_lag", "direct_market_window"]
WEATHER_FEATURE_GROUPS = [
    "direct_time",
    "direct_price_lag",
    "direct_market_window",
    "direct_weather_window",
]
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
            "default": self._prepare_data(hour, DEFAULT_FEATURE_GROUPS),
        }
        if hour in WEAK_HOURS:
            data_by_variant["weather"] = self._prepare_data(hour, WEATHER_FEATURE_GROUPS)

        base_params = get_default_params(self.model_type, hour=hour)
        candidates = self._build_candidates(base_params, hour in WEAK_HOURS)
        rows = []

        for feature_variant, data in data_by_variant.items():
            for idx, params in enumerate(candidates, start=1):
                start = time.time()
                row = {
                    "hour": hour,
                    "status": "success",
                    "candidate_id": idx,
                    "feature_variant": feature_variant,
                    "feature_count": len(data["feature_cols"]),
                    "test_period": data["split_info"]["test_period"],
                    "n_train": data["split_info"]["n_train"],
                    "n_test": data["split_info"]["n_test"],
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "is_default_params": idx == 1,
                }
                try:
                    model = create_model(self.model_type, params)
                    model.fit(data["X_train"], data["y_train"])
                    pred = model.predict(data["X_test"])
                    row.update(
                        {
                            "mae": calculate_mae(data["y_test"], pred),
                            "rmse": calculate_rmse(data["y_test"], pred),
                            "smape": calculate_smape(data["y_test"], pred),
                            "training_time": time.time() - start,
                        }
                    )
                except Exception as exc:
                    row.update({"status": "failed", "error": str(exc), "training_time": time.time() - start})
                rows.append(row)

        return rows

    def _prepare_data(self, hour: int, feature_groups: Sequence[str]) -> Dict[str, Any]:
        features_df, target_col = self.engineer.load_features(hour)
        candidate_features = [col for col in features_df.columns if col not in [target_col, "预测日期"]]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = self.feature_selector.select_features_from_groups(list(feature_groups), numeric_features)
        split = split_by_months(features_df, "预测日期", self.test_months)

        return {
            "X_train": features_df.loc[split.train_mask, feature_cols].reset_index(drop=True),
            "y_train": features_df.loc[split.train_mask, target_col].to_numpy(),
            "X_test": features_df.loc[split.test_mask, feature_cols].reset_index(drop=True),
            "y_test": features_df.loc[split.test_mask, target_col].to_numpy(),
            "feature_cols": feature_cols,
            "split_info": split.to_dict(),
        }

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
        safe_period = test_period.replace(",", "_")
        hour_suffix = ""
        if sorted(requested_hours) != list(range(24)):
            hour_suffix = "_" + "-".join(f"H{hour:02d}" for hour in sorted(requested_hours))
        log_dir = self.config.get_result_path("logs") / "direct" / self.model_type
        log_dir.mkdir(parents=True, exist_ok=True)

        csv_path = log_dir / f"probe_optimization_{safe_period}{hour_suffix}.csv"
        results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        best_df = success_df.sort_values(["hour", "smape"]).groupby("hour", as_index=False).head(1)
        default_df = success_df[success_df["is_default_params"] & (success_df["feature_variant"] == "default")]

        summary = self._build_summary(best_df, default_df, requested_hours, test_period)
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
    ) -> Dict[str, Any]:
        baseline_smape = float(default_df["smape"].mean())
        best_smape = float(best_df["smape"].mean())
        baseline_under20 = int((default_df["smape"] < 20).sum())
        best_under20 = int((best_df["smape"] < 20).sum())

        requested_set = set(requested_hours)
        midday_hours = sorted(requested_set & set(range(8, 17)))
        if midday_hours:
            baseline_midday = float(default_df[default_df["hour"].isin(midday_hours)]["smape"].mean())
            best_midday = float(best_df[best_df["hour"].isin(midday_hours)]["smape"].mean())
        else:
            baseline_midday = None
            best_midday = None

        acceptance_reasons = []
        if best_smape < 27.0:
            acceptance_reasons.append("overall_smape_below_27")
        if best_under20 >= baseline_under20 + 2:
            acceptance_reasons.append("under20_hours_plus_2")
        if (
            baseline_midday is not None
            and best_midday is not None
            and baseline_midday - best_midday >= 3.0
            and best_smape <= baseline_smape
        ):
            acceptance_reasons.append("midday_smape_down_3_without_overall_regression")

        best_params = {}
        for row in best_df.sort_values("hour").itertuples(index=False):
            best_params[f"H{int(row.hour):02d}"] = {
                "smape": float(row.smape),
                "feature_variant": row.feature_variant,
                "params": json.loads(row.params),
            }

        return {
            "model_type": self.model_type,
            "test_period": test_period,
            "hours": requested_hours,
            "baseline_smape": baseline_smape,
            "best_smape": best_smape,
            "baseline_under20_hours": baseline_under20,
            "best_under20_hours": best_under20,
            "baseline_midday_smape": baseline_midday,
            "best_midday_smape": best_midday,
            "accepted": bool(acceptance_reasons),
            "acceptance_reasons": acceptance_reasons,
            "best_params_by_hour": best_params,
        }


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
    parser = argparse.ArgumentParser(description="Optimize LightGBM sMAPE probe parameters on monthly holdout")
    parser.add_argument("--model", default="lightgbm_smape_probe_v2", choices=list_model_types())
    parser.add_argument("--hours", type=int, nargs="+", default=None)
    parser.add_argument("--test-months", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=80)
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
        local_alpha_radius=args.local_alpha_radius,
        local_alpha_step=args.local_alpha_step,
        broad_alpha_step=args.broad_alpha_step,
    ).optimize(args.hours)
    ok = results[results["status"] == "success"]
    best = ok.sort_values(["hour", "smape"]).groupby("hour", as_index=False).head(1)
    print(best[["hour", "feature_variant", "smape", "mae", "rmse"]].to_string(index=False))
    print(f"\nBest-combined average sMAPE={best['smape'].mean():.2f}%")


if __name__ == "__main__":
    main()