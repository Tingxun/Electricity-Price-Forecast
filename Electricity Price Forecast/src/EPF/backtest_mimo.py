"""Expanding-window backtests for MIMO models."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config
from feature_engineering_mimo import MimoFeatureEngineer
from model_tcn_mimo import TcnMimoNet, resolve_device
from train_mimo import MimoTrainer, SUPPORTED_MIMO_MODELS
from utils.data_split import list_rolling_months
from utils.evaluation import prediction_rows_from_wide, save_prediction_report, summarize_predictions
from utils.metrics import calculate_accuracy_rate, calculate_mae, calculate_rmse, calculate_smape


logger = logging.getLogger(__name__)

FORWARD_DEFAULT_START_MONTH = "2025-03"
FORWARD_DEFAULT_END_MONTH = "2025-06"


class MimoBacktester:
    def __init__(
        self,
        config: Config,
        model_type: str = "tcn_mimo",
        min_train_months: int = 3,
        start_month: Optional[str] = FORWARD_DEFAULT_START_MONTH,
        end_month: Optional[str] = FORWARD_DEFAULT_END_MONTH,
        retrain_frequency: str = "monthly",
        model_config: Optional[Dict[str, Any]] = None,
    ):
        if model_type not in SUPPORTED_MIMO_MODELS:
            raise ValueError(f"unsupported MIMO model_type: {model_type}")
        if retrain_frequency not in {"monthly", "weekly"}:
            raise ValueError("retrain_frequency must be 'monthly' or 'weekly'")
        self.config = config
        self.model_type = model_type
        self.min_train_months = min_train_months
        self.start_month = start_month
        self.end_month = end_month
        self.retrain_frequency = retrain_frequency
        self.trainer = MimoTrainer(config, model_type, model_config=model_config)
        self.output_prefix = "weekly_backtest" if retrain_frequency == "weekly" else "monthly_backtest"
        self.rolling_mode = "expanding_forward_weekly" if retrain_frequency == "weekly" else "expanding_forward"
        self.prediction_rows: List[pd.DataFrame] = []

    def run(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        samples = MimoFeatureEngineer(lookback_days=self.trainer.model_config.lookback_days).load_features()
        ref_df = pd.DataFrame({"预测日期": pd.to_datetime(samples["dates"])})
        months = list_rolling_months(ref_df, "预测日期", self.min_train_months, self.start_month, self.end_month)
        rows = []
        for month in months:
            if self.retrain_frequency == "weekly":
                for week_id, week_start, week_end in self._weekly_windows(samples["dates"], month):
                    rows.append(self._run_window(samples, month, week_id, week_start, week_end, hours))
            else:
                period = pd.Period(month, freq="M")
                rows.append(
                    self._run_window(
                        samples,
                        month,
                        None,
                        period.start_time.strftime("%Y-%m-%d"),
                        period.end_time.strftime("%Y-%m-%d"),
                        hours,
                    )
                )
        results_df = pd.DataFrame(rows)
        self._save_outputs(results_df)
        return results_df

    def _run_window(
        self,
        samples: Dict[str, Any],
        test_month: str,
        week_id: Optional[str],
        week_start: str,
        week_end: str,
        hours: Optional[List[int]],
    ) -> Dict[str, Any]:
        start = time.time()
        dates = pd.to_datetime(pd.Series(samples["dates"]))
        start_ts = pd.Timestamp(week_start)
        end_ts = pd.Timestamp(week_end)
        train_idx = np.where((dates < start_ts).to_numpy())[0]
        test_idx = np.where(((dates >= start_ts) & (dates <= end_ts)).to_numpy())[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"invalid MIMO backtest window {week_start} to {week_end}")
        pred = self._train_predict(samples, train_idx, test_idx)
        actual = samples["targets"][test_idx]
        pred_df = prediction_rows_from_wide(
            samples["dates"][test_idx],
            actual,
            pred,
            rolling_mode=self.rolling_mode,
            week_id=week_id,
            week_start=week_start,
            week_end=week_end,
        )
        if hours is not None:
            metric_df = pred_df[pred_df["hour"].isin(hours)]
        else:
            metric_df = pred_df
        self.prediction_rows.append(pred_df)
        return {
            "rolling_mode": self.rolling_mode,
            "test_month": test_month,
            "week_id": week_id,
            "week_start": week_start,
            "week_end": week_end,
            "status": "success",
            "train_start": str(samples["dates"][train_idx][0]),
            "train_end": str(samples["dates"][train_idx][-1]),
            "test_start": str(samples["dates"][test_idx][0]),
            "test_end": str(samples["dates"][test_idx][-1]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx) * (len(hours) if hours else 24)),
            "mae": calculate_mae(metric_df["actual"], metric_df["pred"]),
            "rmse": calculate_rmse(metric_df["actual"], metric_df["pred"]),
            "smape": calculate_smape(metric_df["actual"], metric_df["pred"]),
            "acc_rate": calculate_accuracy_rate(metric_df["actual"], metric_df["pred"], threshold=20.0),
            "training_time": time.time() - start,
        }

    def _train_predict(self, samples: Dict[str, Any], train_window_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        train_idx, val_idx = MimoTrainer._train_val_split(samples["dates"], train_window_idx)
        scalers = self.trainer._fit_scalers(samples, train_idx)
        tensors = MimoTrainer._build_tensors(samples, scalers)
        device = resolve_device(self.trainer.model_config.device)
        torch.manual_seed(self.trainer.model_config.seed)
        np.random.seed(self.trainer.model_config.seed)
        model = TcnMimoNet(samples["target_exog"].shape[2], self.trainer.model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.trainer.model_config.lr,
            weight_decay=self.trainer.model_config.weight_decay,
        )
        loader = DataLoader(
            TensorDataset(tensors["price"][train_idx], tensors["exog"][train_idx], tensors["y_scaled"][train_idx]),
            batch_size=self.trainer.model_config.batch_size,
            shuffle=True,
        )
        best_state = None
        best_score = float("inf")
        wait = 0
        for _epoch in range(1, self.trainer.model_config.epochs + 1):
            self.trainer._train_one_epoch(model, loader, optimizer, device, scalers)
            val_pred = MimoTrainer._predict_scaled(model, tensors, val_idx, device)
            val_pred = MimoTrainer._clip_predictions(MimoTrainer._inverse_target(val_pred, scalers), scalers)
            score = MimoTrainer._robust_score(samples["dates"][val_idx], samples["targets"][val_idx], val_pred)
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.trainer.model_config.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        pred_scaled = MimoTrainer._predict_scaled(model, tensors, test_idx, device)
        return MimoTrainer._clip_predictions(MimoTrainer._inverse_target(pred_scaled, scalers), scalers)

    def _save_outputs(self, results_df: pd.DataFrame) -> None:
        log_dir = self.config.get_result_path("logs") / "mimo" / self.model_type
        log_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(log_dir / f"{self.output_prefix}.csv", index=False, encoding="utf-8-sig")
        prediction_df = pd.concat(self.prediction_rows, ignore_index=True) if self.prediction_rows else pd.DataFrame()
        metadata = {
            "strategy": "mimo",
            "model_type": self.model_type,
            "rolling_mode": self.rolling_mode,
            "retrain_frequency": self.retrain_frequency,
            "start_month": self.start_month,
            "end_month": self.end_month,
            "n_rows": int(results_df.shape[0]),
        }
        save_prediction_report(prediction_df, log_dir, self.output_prefix, metadata=metadata)

    @staticmethod
    def _weekly_windows(dates: np.ndarray, test_month: str) -> List[Tuple[str, str, str]]:
        date_series = pd.to_datetime(pd.Series(dates))
        month = pd.Period(test_month, freq="M")
        month_dates = date_series[date_series.dt.to_period("M") == month]
        windows = []
        start = month_dates.min().normalize()
        end = month_dates.max().normalize()
        idx = 1
        while start <= end:
            window_end = min(start + pd.Timedelta(days=6), end)
            windows.append((f"W{idx:02d}", start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
            start = window_end + pd.Timedelta(days=1)
            idx += 1
        return windows


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "backtest_mimo.log", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MIMO models")
    parser.add_argument("--model", default="tcn_mimo", choices=SUPPORTED_MIMO_MODELS)
    parser.add_argument("--hours", type=int, nargs="+", default=None)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--start-month", default=FORWARD_DEFAULT_START_MONTH)
    parser.add_argument("--end-month", default=FORWARD_DEFAULT_END_MONTH)
    parser.add_argument("--retrain-frequency", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    overrides = {k: v for k, v in {"epochs": args.epochs, "patience": args.patience, "device": args.device}.items() if v is not None}
    results = MimoBacktester(
        Config(),
        args.model,
        min_train_months=args.min_train_months,
        start_month=args.start_month,
        end_month=args.end_month,
        retrain_frequency=args.retrain_frequency,
        model_config=overrides,
    ).run(args.hours)
    print(results.to_string(index=False))
    ok = results[results["status"] == "success"]
    print(f"\nAverage sMAPE={float((ok['smape'] * ok['n_test']).sum() / ok['n_test'].sum()):.2f}%")


if __name__ == "__main__":
    main()
