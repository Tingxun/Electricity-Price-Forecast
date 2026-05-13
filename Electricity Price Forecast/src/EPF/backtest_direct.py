"""
Expanding-window monthly or weekly-retrain backtest for Direct forecasting models.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

from .config import Config
from .utils.data_split import list_rolling_months, split_by_months
from .feature_engineering_direct import DirectFeatureEngineer
from .feature_selector import FeatureSelector
from .model_factory import create_model, get_default_params, get_param_space, list_model_types
from .train_direct import DirectTrainer
from .utils.metrics import calculate_accuracy_rate, calculate_mae, calculate_rmse, calculate_sape, calculate_smape
from .utils.evaluation import summarize_predictions


logger = logging.getLogger(__name__)

FORWARD_DEFAULT_START_MONTH = "2025-03"
FORWARD_DEFAULT_END_MONTH = "2025-06"
ROLLING_MODE = "expanding_forward"
WEEKLY_ROLLING_MODE = "expanding_forward_weekly"
DATE_COL = "\u9884\u6d4b\u65e5\u671f"


class DirectMonthlyBacktester:
    """Run expanding-window backtests without overwriting saved models."""

    def __init__(
        self,
        config: Config,
        model_type: str,
        n_iter: int,
        cv_folds: int,
        min_train_months: int,
        start_month: Optional[str] = FORWARD_DEFAULT_START_MONTH,
        end_month: Optional[str] = FORWARD_DEFAULT_END_MONTH,
        retrain_frequency: str = "monthly",
    ):
        if retrain_frequency not in {"monthly", "weekly"}:
            raise ValueError("retrain_frequency must be 'monthly' or 'weekly'")
        self.config = config
        self.model_type = model_type
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.min_train_months = min_train_months
        self.start_month = start_month
        self.end_month = end_month
        self.retrain_frequency = retrain_frequency
        self.feature_selector = FeatureSelector()
        self.engineer = DirectFeatureEngineer()
        self.rolling_mode = WEEKLY_ROLLING_MODE if retrain_frequency == "weekly" else ROLLING_MODE
        self.output_prefix = "weekly_backtest" if retrain_frequency == "weekly" else "monthly_backtest"
        self.prediction_rows: List[Dict[str, Any]] = []

    def run(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        if hours is None:
            hours = list(range(24))

        reference_df, _ = self.engineer.load_features(hours[0])
        months = list_rolling_months(
            reference_df,
            DATE_COL,
            min_train_months=self.min_train_months,
            start_month=self.start_month,
            end_month=self.end_month,
        )
        if not months:
            raise ValueError("没有可回测月份，请降低 --min-train-months 或检查 --start-month/--end-month")

        rows = []
        for test_month in months:
            logger.info("开始回测月份 %s", test_month)
            if self.retrain_frequency == "weekly":
                windows = self._weekly_windows_for_month(reference_df, DATE_COL, test_month)
                for week_id, week_start, week_end in windows:
                    logger.info("开始周度回测 %s %s %s-%s", test_month, week_id, week_start, week_end)
                    for hour in hours:
                        try:
                            rows.append(self._run_one_hour(test_month, hour, week_id, week_start, week_end))
                        except Exception as exc:
                            logger.exception("回测失败: month=%s, %s, H%02d", test_month, week_id, hour)
                            rows.append(
                                {
                                    "test_month": test_month,
                                    "week_id": week_id,
                                    "week_start": week_start,
                                    "week_end": week_end,
                                    "hour": hour,
                                    "status": "failed",
                                    "error": str(exc),
                                }
                            )
            else:
                for hour in hours:
                    try:
                        rows.append(self._run_one_hour(test_month, hour))
                    except Exception as exc:
                        logger.exception("回测失败: month=%s, H%02d", test_month, hour)
                        rows.append({"test_month": test_month, "hour": hour, "status": "failed", "error": str(exc)})

        results_df = pd.DataFrame(rows)
        self._save_outputs(results_df)
        return results_df

    def _run_one_hour(
        self,
        test_month: str,
        hour: int,
        week_id: Optional[str] = None,
        week_start: Optional[str] = None,
        week_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        data = self._prepare_data(hour, test_month, week_start=week_start, week_end=week_end)
        selected_structure = "fixed_default"
        selected_feature_groups: List[str] = []
        if self.model_type == "lightgbm_auto":
            helper = DirectTrainer(
                config=self.config,
                model_type=self.model_type,
                n_iter=self.n_iter,
                cv_folds=self.cv_folds,
                test_months=[test_month],
            )
            selection_info = helper._select_auto_structure(data, hour)
            data["feature_cols"] = selection_info["feature_cols"]
            data["X_train"] = data["full_X_train"][data["feature_cols"]].reset_index(drop=True)
            data["X_test"] = data["full_X_test"][data["feature_cols"]].reset_index(drop=True)
            best_params, cv_smape, _ = helper._search_best_params(
                data["X_train"],
                data["y_train"],
                hour=hour,
                random_state=42 + hour,
                dates=data["train_dates"],
                base_params=selection_info["selected_params"],
            )
            selected_structure = selection_info["selected_structure"]
            selected_feature_groups = selection_info["selected_feature_groups"]
        else:
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
                    "week_id": week_id,
                    "week_start": week_start,
                    "week_end": week_end,
                    "hour": hour,
                    DATE_COL: pd.Timestamp(date).strftime("%Y-%m-%d"),
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
            "week_id": week_id,
            "week_start": week_start,
            "week_end": week_end,
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
            "selected_structure": selected_structure,
            "selected_feature_groups": ",".join(selected_feature_groups),
            "best_params": json.dumps(best_params, ensure_ascii=False),
        }

    def _prepare_data(
        self,
        hour: int,
        test_month: str,
        week_start: Optional[str] = None,
        week_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        features_df, target_col = self.engineer.load_features(hour)
        candidate_features = [col for col in features_df.columns if col not in [target_col, DATE_COL]]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = self.feature_selector.select_features_for_model(self.model_type, numeric_features, hour=hour)
        feature_info = self.feature_selector.get_model_feature_info(self.model_type)
        if week_start is not None or week_end is not None:
            if week_start is None or week_end is None:
                raise ValueError("week_start and week_end must be provided together")
            train_mask, test_mask, split_info = self._split_by_week_window(
                features_df,
                test_month=test_month,
                week_start=week_start,
                week_end=week_end,
            )
        else:
            split = split_by_months(features_df, DATE_COL, [test_month])
            train_mask = split.train_mask
            test_mask = split.test_mask
            split_info = split.to_dict()

        X_train = features_df.loc[train_mask, feature_cols].reset_index(drop=True)
        y_train = features_df.loc[train_mask, target_col].to_numpy()
        X_test = features_df.loc[test_mask, feature_cols].reset_index(drop=True)
        y_test = features_df.loc[test_mask, target_col].to_numpy()
        full_X_train = features_df.loc[train_mask, numeric_features].reset_index(drop=True)
        full_X_test = features_df.loc[test_mask, numeric_features].reset_index(drop=True)
        train_dates = features_df.loc[train_mask, DATE_COL].reset_index(drop=True)
        test_dates = features_df.loc[test_mask, DATE_COL].reset_index(drop=True)

        if feature_info.get("normalize", False):
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "full_X_train": full_X_train,
            "full_X_test": full_X_test,
            "numeric_features": numeric_features,
            "feature_cols": feature_cols,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "split_info": split_info,
        }

    @staticmethod
    def _weekly_windows_for_month(
        df: pd.DataFrame,
        date_col: str,
        test_month: str,
    ) -> List[Tuple[str, str, str]]:
        dates = pd.to_datetime(df[date_col])
        month = pd.Period(test_month, freq="M")
        month_dates = dates[dates.dt.to_period("M") == month]
        if month_dates.empty:
            raise ValueError(f"no dates found for test month {test_month}")

        month_start = month_dates.min().normalize()
        month_end = month_dates.max().normalize()
        windows: List[Tuple[str, str, str]] = []
        window_start = month_start
        idx = 1
        while window_start <= month_end:
            window_end = min(window_start + pd.Timedelta(days=6), month_end)
            windows.append(
                (
                    f"W{idx:02d}",
                    window_start.strftime("%Y-%m-%d"),
                    window_end.strftime("%Y-%m-%d"),
                )
            )
            window_start = window_end + pd.Timedelta(days=1)
            idx += 1
        return windows

    @staticmethod
    def _split_by_week_window(
        df: pd.DataFrame,
        test_month: str,
        week_start: str,
        week_end: str,
    ) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
        dates = pd.to_datetime(df[DATE_COL])
        start = pd.Timestamp(week_start)
        end = pd.Timestamp(week_end)
        month = pd.Period(test_month, freq="M")

        train_mask = dates < start
        test_mask = (dates >= start) & (dates <= end) & (dates.dt.to_period("M") == month)
        if not train_mask.any():
            raise ValueError(f"no training data before weekly window {week_start}")
        if not test_mask.any():
            raise ValueError(f"no test data in weekly window {week_start} to {week_end}")

        train_dates = dates[train_mask]
        test_dates = dates[test_mask]
        split_info = {
            "split_strategy": "weekly_retrain",
            "test_months": [test_month],
            "test_period": test_month,
            "train_start": train_dates.min().strftime("%Y-%m-%d"),
            "train_end": train_dates.max().strftime("%Y-%m-%d"),
            "test_start": test_dates.min().strftime("%Y-%m-%d"),
            "test_end": test_dates.max().strftime("%Y-%m-%d"),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        return train_mask, test_mask, split_info

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

        results_df.to_csv(log_dir / f"{self.output_prefix}.csv", index=False, encoding="utf-8-sig")
        prediction_df = pd.DataFrame(self.prediction_rows)
        if not prediction_df.empty:
            prediction_df.to_csv(log_dir / f"{self.output_prefix}_predictions.csv", index=False, encoding="utf-8-sig")
            shared_summary = summarize_predictions(prediction_df)
            shared_summary["hour_summary"].to_csv(log_dir / f"{self.output_prefix}_hour_summary.csv", index=False, encoding="utf-8-sig")
            shared_summary["bucket_summary"].to_csv(log_dir / f"{self.output_prefix}_bucket_summary.csv", index=False, encoding="utf-8-sig")
        else:
            shared_summary = {"overall": {}}

        success_df = results_df[results_df["status"] == "success"].copy()
        summary_source_df = success_df
        if not prediction_df.empty:
            summary_source_df = (
                prediction_df.groupby(["test_month", "hour"], as_index=False)
                .agg(
                    mae=("abs_error", "mean"),
                    rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                    smape=("sape", "mean"),
                    acc_rate=("sape", lambda s: float((s < 20.0).mean() * 100.0)),
                )
            )
        month_summary = (
            summary_source_df.groupby("test_month", as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                smape=("smape", "mean"),
                hourly_acc_rate=("acc_rate", "mean"),
                under20_hours=("smape", lambda s: int((s < 20).sum())),
            )
        )
        month_summary["rolling_mode"] = self.rolling_mode
        month_summary["midday_smape"] = month_summary["test_month"].map(self._band_smape(summary_source_df, range(8, 16)))
        month_summary["non_midday_smape"] = month_summary["test_month"].map(
            self._band_smape(summary_source_df, [*range(0, 8), *range(16, 24)])
        )
        month_summary["worst_hours"] = month_summary["test_month"].map(self._worst_hours(summary_source_df))
        if not prediction_df.empty:
            prediction_metrics = (
                prediction_df.groupby("test_month", as_index=False)
                .agg(
                    pred_mae=("abs_error", "mean"),
                    pred_rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                    pred_smape=("sape", "mean"),
                )
            )
            month_summary = month_summary.merge(prediction_metrics, on="test_month", how="left")
            month_summary["mae"] = month_summary["pred_mae"].fillna(month_summary["mae"])
            month_summary["rmse"] = month_summary["pred_rmse"].fillna(month_summary["rmse"])
            month_summary["smape"] = month_summary["pred_smape"].fillna(month_summary["smape"])
            month_summary = month_summary.drop(columns=["pred_mae", "pred_rmse", "pred_smape"])
            monthly_acc = prediction_df.groupby("test_month")["sape"].apply(lambda s: float((s < 20.0).mean() * 100.0))
            month_summary["monthly_acc_rate"] = month_summary["test_month"].map(monthly_acc)
        else:
            month_summary["monthly_acc_rate"] = month_summary["hourly_acc_rate"]
        month_summary["smape_below_30"] = month_summary["smape"] < 30.0
        month_summary["monthly_acc_rate_ge_50"] = month_summary["monthly_acc_rate"] >= 50.0
        month_summary.to_csv(log_dir / f"{self.output_prefix}_summary.csv", index=False, encoding="utf-8-sig")

        summary = {
            "model_type": self.model_type,
            "rolling_mode": self.rolling_mode,
            "retrain_frequency": self.retrain_frequency,
            "start_month": self.start_month,
            "end_month": self.end_month,
            "n_months": int(month_summary.shape[0]),
            "n_rows": int(success_df.shape[0]),
            "overall_mae": float(prediction_df["abs_error"].mean())
            if not prediction_df.empty
            else (float(success_df["mae"].mean()) if not success_df.empty else None),
            "overall_rmse": float(np.sqrt(np.mean(np.square(prediction_df["error"]))))
            if not prediction_df.empty
            else (float(success_df["rmse"].mean()) if not success_df.empty else None),
            "overall_smape": float(prediction_df["sape"].mean())
            if not prediction_df.empty
            else (float(success_df["smape"].mean()) if not success_df.empty else None),
            "overall_acc_rate": float((prediction_df["sape"] < 20.0).mean() * 100.0) if not prediction_df.empty else None,
            "avg_under20_hours": float(month_summary["under20_hours"].mean()) if not month_summary.empty else None,
            "months_below_30": int(month_summary["smape_below_30"].sum()) if not month_summary.empty else 0,
            "months_acc_rate_ge_50": int(month_summary["monthly_acc_rate_ge_50"].sum()) if not month_summary.empty else 0,
            "failed_months": month_summary.loc[
                ~(month_summary["smape_below_30"] & month_summary["monthly_acc_rate_ge_50"]),
                ["test_month", "smape", "monthly_acc_rate", "worst_hours"],
            ].to_dict("records") if not month_summary.empty else [],
        }
        summary.update({k: v for k, v in shared_summary.get("overall", {}).items() if v is not None})
        with open(log_dir / f"{self.output_prefix}_overall.json", "w", encoding="utf-8") as f:
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
    parser.add_argument(
        "--retrain-frequency",
        choices=["monthly", "weekly"],
        default="monthly",
        help="monthly 表示每个测试月训练一次；weekly 表示测试月内每 7 天重新训练一次",
    )
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
        retrain_frequency=args.retrain_frequency,
    ).run(args.hours)
    print(results.to_string(index=False))
    ok = results[results["status"] == "success"]
    if not ok.empty and "n_test" in ok:
        avg_smape = float((ok["smape"] * ok["n_test"]).sum() / ok["n_test"].sum())
    else:
        avg_smape = float(ok["smape"].mean())
    print(f"\nAverage sMAPE={avg_smape:.2f}%")


if __name__ == "__main__":
    main()
