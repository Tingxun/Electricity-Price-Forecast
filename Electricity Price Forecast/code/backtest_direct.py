"""
Expanding-window monthly backtest for Direct forecasting models.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config
from data_split import list_rolling_months, split_by_months
from feature_engineering_direct import DirectFeatureEngineer
from feature_selector import FeatureSelector
from model_factory import create_model, get_default_params, get_param_space, list_model_types
from utils.metrics import calculate_accuracy_rate, calculate_mae, calculate_rmse, calculate_sape, calculate_smape


logger = logging.getLogger(__name__)

FORWARD_DEFAULT_START_MONTH = "2025-03"
FORWARD_DEFAULT_END_MONTH = "2025-06"
ROLLING_MODE = "expanding_forward"


class DirectMonthlyBacktester:
    """Run expanding-window monthly backtests without overwriting saved models."""

    def __init__(
        self,
        config: Config,
        model_type: str,
        n_iter: int,
        cv_folds: int,
        min_train_months: int,
        start_month: Optional[str] = FORWARD_DEFAULT_START_MONTH,
        end_month: Optional[str] = FORWARD_DEFAULT_END_MONTH,
    ):
        self.config = config
        self.model_type = model_type
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.min_train_months = min_train_months
        self.start_month = start_month
        self.end_month = end_month
        self.feature_selector = FeatureSelector()
        self.engineer = DirectFeatureEngineer()
        self.rolling_mode = ROLLING_MODE
        self.prediction_rows: List[Dict[str, Any]] = []

    def run(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        if hours is None:
            hours = list(range(24))

        reference_df, _ = self.engineer.load_features(hours[0])
        months = list_rolling_months(
            reference_df,
            "预测日期",
            min_train_months=self.min_train_months,
            start_month=self.start_month,
            end_month=self.end_month,
        )
        if not months:
            raise ValueError("没有可回测月份，请降低 --min-train-months 或检查 --start-month/--end-month")

        rows = []
        for test_month in months:
            logger.info("开始回测月份 %s", test_month)
            for hour in hours:
                try:
                    rows.append(self._run_one_hour(test_month, hour))
                except Exception as exc:
                    logger.exception("回测失败: month=%s, H%02d", test_month, hour)
                    rows.append({"test_month": test_month, "hour": hour, "status": "failed", "error": str(exc)})

        results_df = pd.DataFrame(rows)
        self._save_outputs(results_df)
        return results_df

    def _run_one_hour(self, test_month: str, hour: int) -> Dict[str, Any]:
        start = time.time()
        data = self._prepare_data(hour, test_month)
        best_params, cv_smape = self._search_best_params(data["X_train"], data["y_train"], hour, 42 + hour)

        model = create_model(self.model_type, best_params)
        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])
        sape = calculate_sape(data["y_test"], y_pred)

        for date, actual, pred, sape_value in zip(data["test_dates"], data["y_test"], y_pred, sape):
            self.prediction_rows.append(
                {
                    "rolling_mode": self.rolling_mode,
                    "test_month": test_month,
                    "hour": hour,
                    "预测日期": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "actual": float(actual),
                    "pred": float(pred),
                    "error": float(pred - actual),
                    "abs_error": float(abs(pred - actual)),
                    "sape": float(sape_value),
                    "actual_price_bucket": self._price_bucket(actual),
                }
            )

        return {
            "rolling_mode": self.rolling_mode,
            "test_month": test_month,
            "hour": hour,
            "status": "success",
            "train_start": data["split_info"]["train_start"],
            "train_end": data["split_info"]["train_end"],
            "test_start": data["split_info"]["test_start"],
            "test_end": data["split_info"]["test_end"],
            "n_train": data["split_info"]["n_train"],
            "n_test": data["split_info"]["n_test"],
            "best_cv_smape": cv_smape,
            "mae": calculate_mae(data["y_test"], y_pred),
            "rmse": calculate_rmse(data["y_test"], y_pred),
            "smape": calculate_smape(data["y_test"], y_pred),
            "acc_rate": calculate_accuracy_rate(data["y_test"], y_pred, threshold=20.0),
            "training_time": time.time() - start,
            "best_params": json.dumps(best_params, ensure_ascii=False),
        }

    def _prepare_data(self, hour: int, test_month: str) -> Dict[str, Any]:
        features_df, target_col = self.engineer.load_features(hour)
        candidate_features = [col for col in features_df.columns if col not in [target_col, "预测日期"]]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = self.feature_selector.select_features_for_model(self.model_type, numeric_features, hour=hour)
        feature_info = self.feature_selector.get_model_feature_info(self.model_type)
        split = split_by_months(features_df, "预测日期", [test_month])

        X_train = features_df.loc[split.train_mask, feature_cols].reset_index(drop=True)
        y_train = features_df.loc[split.train_mask, target_col].to_numpy()
        X_test = features_df.loc[split.test_mask, feature_cols].reset_index(drop=True)
        y_test = features_df.loc[split.test_mask, target_col].to_numpy()
        train_dates = features_df.loc[split.train_mask, "预测日期"].reset_index(drop=True)
        test_dates = features_df.loc[split.test_mask, "预测日期"].reset_index(drop=True)

        if feature_info.get("normalize", False):
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "split_info": split.to_dict(),
        }

    @staticmethod
    def _price_bucket(value: float) -> str:
        value = float(value)
        if value <= 0:
            return "zero"
        if value <= 20:
            return "near_zero"
        if value <= 80:
            return "low_20_80"
        if value <= 200:
            return "mid_80_200"
        return "high_200_plus"

    def _search_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        hour: int,
        random_state: int,
    ) -> Tuple[Dict[str, Any], float]:
        default_params = get_default_params(self.model_type, hour=hour)
        param_space = get_param_space(self.model_type)
        if self.n_iter <= 0 or not param_space:
            return default_params, self._cross_val_smape(default_params, X_train, y_train)

        best_params = default_params
        best_score = float("inf")
        for params in ParameterSampler(param_space, n_iter=self.n_iter, random_state=random_state):
            try:
                score = self._cross_val_smape(params, X_train, y_train)
            except Exception as exc:
                logger.warning("参数组合失败: %s", exc)
                continue
            if score < best_score:
                best_score = score
                best_params = dict(params)

        if not np.isfinite(best_score):
            best_score = self._cross_val_smape(best_params, X_train, y_train)
        return best_params, best_score

    def _cross_val_smape(self, params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> float:
        scores = []
        for train_idx, val_idx in self._time_series_folds(len(X), self.cv_folds):
            model = create_model(self.model_type, params)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(calculate_smape(y[val_idx], pred))
        return float(np.mean(scores))

    @staticmethod
    def _time_series_folds(n_samples: int, cv_folds: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if n_samples < cv_folds + 2:
            raise ValueError("样本量不足，无法进行时间序列交叉验证")

        for fold in range(cv_folds):
            train_end = int(n_samples * (fold + 1) / (cv_folds + 1))
            val_end = int(n_samples * (fold + 2) / (cv_folds + 1))
            if train_end == 0 or val_end <= train_end:
                continue
            yield np.arange(0, train_end), np.arange(train_end, val_end)

    def _save_outputs(self, results_df: pd.DataFrame) -> None:
        log_dir = self.config.get_result_path("logs") / "direct" / self.model_type
        log_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(log_dir / "monthly_backtest.csv", index=False, encoding="utf-8-sig")
        prediction_df = pd.DataFrame(self.prediction_rows)
        if not prediction_df.empty:
            prediction_df.to_csv(log_dir / "monthly_backtest_predictions.csv", index=False, encoding="utf-8-sig")

        success_df = results_df[results_df["status"] == "success"].copy()
        month_summary = (
            success_df.groupby("test_month", as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                smape=("smape", "mean"),
                hourly_acc_rate=("acc_rate", "mean"),
                under20_hours=("smape", lambda s: int((s < 20).sum())),
            )
        )
        month_summary["rolling_mode"] = self.rolling_mode
        month_summary["midday_smape"] = month_summary["test_month"].map(self._band_smape(success_df, range(8, 16)))
        month_summary["non_midday_smape"] = month_summary["test_month"].map(
            self._band_smape(success_df, [*range(0, 8), *range(16, 24)])
        )
        month_summary["worst_hours"] = month_summary["test_month"].map(self._worst_hours(success_df))
        if not prediction_df.empty:
            monthly_acc = prediction_df.groupby("test_month")["sape"].apply(lambda s: float((s < 20.0).mean() * 100.0))
            month_summary["monthly_acc_rate"] = month_summary["test_month"].map(monthly_acc)
        else:
            month_summary["monthly_acc_rate"] = month_summary["hourly_acc_rate"]
        month_summary["smape_below_30"] = month_summary["smape"] < 30.0
        month_summary["monthly_acc_rate_ge_50"] = month_summary["monthly_acc_rate"] >= 50.0
        month_summary.to_csv(log_dir / "monthly_backtest_summary.csv", index=False, encoding="utf-8-sig")

        summary = {
            "model_type": self.model_type,
            "rolling_mode": self.rolling_mode,
            "start_month": self.start_month,
            "end_month": self.end_month,
            "n_months": int(month_summary.shape[0]),
            "n_rows": int(success_df.shape[0]),
            "overall_mae": float(success_df["mae"].mean()) if not success_df.empty else None,
            "overall_rmse": float(success_df["rmse"].mean()) if not success_df.empty else None,
            "overall_smape": float(success_df["smape"].mean()) if not success_df.empty else None,
            "overall_acc_rate": float((prediction_df["sape"] < 20.0).mean() * 100.0) if not prediction_df.empty else None,
            "avg_under20_hours": float(month_summary["under20_hours"].mean()) if not month_summary.empty else None,
            "months_below_30": int(month_summary["smape_below_30"].sum()) if not month_summary.empty else 0,
            "months_acc_rate_ge_50": int(month_summary["monthly_acc_rate_ge_50"].sum()) if not month_summary.empty else 0,
            "failed_months": month_summary.loc[
                ~(month_summary["smape_below_30"] & month_summary["monthly_acc_rate_ge_50"]),
                ["test_month", "smape", "monthly_acc_rate", "worst_hours"],
            ].to_dict("records") if not month_summary.empty else [],
        }
        with open(log_dir / "monthly_backtest_overall.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _band_smape(results_df: pd.DataFrame, hours: Iterable[int]) -> Dict[str, float]:
        hour_set = set(hours)
        band_df = results_df[results_df["hour"].isin(hour_set)]
        return band_df.groupby("test_month")["smape"].mean().to_dict()

    @staticmethod
    def _worst_hours(results_df: pd.DataFrame, top_n: int = 5) -> Dict[str, str]:
        result = {}
        for month, group in results_df.groupby("test_month"):
            worst = group.sort_values("smape", ascending=False).head(top_n)
            result[month] = ";".join(f"H{int(row.hour):02d}:{float(row.smape):.2f}" for row in worst.itertuples())
        return result


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "backtest_direct.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct 月份滚动回测")
    parser.add_argument("--model", default="lightgbm", choices=list_model_types(), help="基模型类型")
    parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定回测小时")
    parser.add_argument("--n-iter", type=int, default=0, help="每个小时随机搜索次数；0 表示默认参数")
    parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    parser.add_argument("--min-train-months", type=int, default=3, help="开始回测前至少保留的训练月份数")
    parser.add_argument("--start-month", default=FORWARD_DEFAULT_START_MONTH, help="首个测试月份 YYYY-MM")
    parser.add_argument("--end-month", default=FORWARD_DEFAULT_END_MONTH, help="最后测试月份 YYYY-MM")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    results = DirectMonthlyBacktester(
        config=Config(),
        model_type=args.model,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        min_train_months=args.min_train_months,
        start_month=args.start_month,
        end_month=args.end_month,
    ).run(args.hours)
    print(results.to_string(index=False))
    ok = results[results["status"] == "success"]
    print(f"\nAverage sMAPE={ok['smape'].mean():.2f}%")


if __name__ == "__main__":
    main()
