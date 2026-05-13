"""Evaluate saved MIMO models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import Config
from .feature_engineering_mimo import MimoFeatureEngineer
from .model_tcn_mimo import TcnMimoConfig, TcnMimoNet, resolve_device
from .train_mimo import MimoTrainer, SUPPORTED_MIMO_MODELS
from .utils.evaluation import prediction_rows_from_wide, save_prediction_report, summarize_predictions
from .utils.metrics import calculate_accuracy_rate, calculate_mae, calculate_rmse, calculate_smape
from .utils.model_store import safe_period_label


logger = logging.getLogger(__name__)


class MimoEvaluator:
    def __init__(self, config: Config, model_type: str = "tcn_mimo", test_months: Optional[List[str]] = None):
        self.config = config
        self.model_type = model_type
        self.test_months = test_months
        self.model_dir = self._resolve_model_dir()

    def evaluate(self, hours: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        metadata, model = self._load_model()
        samples = MimoFeatureEngineer(lookback_days=metadata["model_config"]["lookback_days"]).load_features()
        split = MimoTrainer._month_split(samples["dates"], self.test_months or metadata.get("test_months"))
        pred = self._predict(model, samples, split["test_idx"], metadata)
        actual = samples["targets"][split["test_idx"]]
        dates = samples["dates"][split["test_idx"]]
        prediction_df = prediction_rows_from_wide(dates, actual, pred)
        if hours is not None:
            prediction_df = prediction_df[prediction_df["hour"].isin(hours)].reset_index(drop=True)

        rows = []
        for hour, group in prediction_df.groupby("hour"):
            rows.append(
                {
                    "hour": int(hour),
                    "mae": calculate_mae(group["actual"], group["pred"]),
                    "rmse": calculate_rmse(group["actual"], group["pred"]),
                    "smape": calculate_smape(group["actual"], group["pred"]),
                    "acc_rate": calculate_accuracy_rate(group["actual"], group["pred"], threshold=20.0),
                    "n_test": int(group.shape[0]),
                    "test_period": split["test_period"],
                }
            )
        results_df = pd.DataFrame(rows).sort_values("hour")
        self._save_outputs(results_df, prediction_df, split["test_period"])
        return results_df, summarize_predictions(prediction_df)["overall"]

    def _save_outputs(self, results_df: pd.DataFrame, prediction_df: pd.DataFrame, test_period: str) -> None:
        period_label = safe_period_label(test_period)
        log_dir = self.config.get_result_path("logs") / "mimo" / self.model_type / period_label
        pred_dir = self.config.get_result_path("predictions") / "mimo" / self.model_type / period_label
        log_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(log_dir / "evaluation_report.csv", index=False, encoding="utf-8-sig")
        results_df.to_json(log_dir / "evaluation_report.json", orient="records", force_ascii=False, indent=2)
        save_prediction_report(
            prediction_df,
            log_dir,
            "evaluation",
            metadata={"strategy": "mimo", "model_type": self.model_type, "test_period": test_period, "model_dir": str(self.model_dir)},
        )
        wide = self._wide_predictions(prediction_df)
        wide.to_csv(pred_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    @staticmethod
    def _wide_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(prediction_df["预测日期"].unique())
        wide = pd.DataFrame({"预测日期": dates})
        for hour in sorted(prediction_df["hour"].unique()):
            hour_df = prediction_df[prediction_df["hour"] == hour][["预测日期", "actual", "pred"]]
            wide = wide.merge(hour_df.rename(columns={"actual": f"actual_H{hour:02d}", "pred": f"pred_H{hour:02d}"}), on="预测日期", how="left")
        return wide

    def _resolve_model_dir(self) -> Path:
        base_dir = self.config.project_root / "saved_models" / "mimo" / self.model_type
        if self.test_months:
            return base_dir / safe_period_label(",".join(self.test_months))
        runs = [path for path in base_dir.iterdir() if path.is_dir() and (path / "metadata.json").exists()] if base_dir.exists() else []
        if not runs:
            return base_dir
        return max(runs, key=lambda path: path.stat().st_mtime)

    def _load_model(self):
        metadata_path = self.model_dir / "metadata.json"
        model_path = self.model_dir / "model.pt"
        if not metadata_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"missing MIMO model artifacts in {self.model_dir}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        config = TcnMimoConfig.from_dict(metadata["model_config"])
        device = resolve_device(config.device)
        model = TcnMimoNet(len(metadata["exog_columns"]), config).to(device)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return metadata, model

    def _predict(self, model, samples: Dict[str, Any], indices: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        scalers = metadata["scalers"]
        tensors = MimoTrainer._build_tensors(samples, scalers)
        config = TcnMimoConfig.from_dict(metadata["model_config"])
        device = resolve_device(config.device)
        pred_scaled = MimoTrainer._predict_scaled(model, tensors, indices, device)
        pred = MimoTrainer._inverse_target(pred_scaled, scalers)
        return MimoTrainer._clip_predictions(pred, scalers)


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "evaluate_mimo.log", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIMO models")
    parser.add_argument("--model", default="tcn_mimo", choices=SUPPORTED_MIMO_MODELS)
    parser.add_argument("--hours", type=int, nargs="+", default=None)
    parser.add_argument("--test-months", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    results, overall = MimoEvaluator(Config(), args.model, args.test_months).evaluate(args.hours)
    print(results.to_string(index=False))
    print(
        f"\nMAE={results['mae'].mean():.4f}, RMSE={results['rmse'].mean():.4f}, "
        f"sMAPE={overall['overall_smape']:.2f}%, AccRate={overall['overall_acc_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()
