"""
Evaluate trained Direct multi-step models on the held-out tail split.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config
from feature_engineering_direct import DirectFeatureEngineer
from feature_selector import FeatureSelector
from model_factory import list_model_types
from utils.metrics import calculate_mae, calculate_rmse, calculate_smape


logger = logging.getLogger(__name__)


class DirectEvaluator:
    """Evaluate one saved Direct model family."""

    def __init__(self, config: Config, model_type: str):
        self.config = config
        self.model_type = model_type
        self.model_dir = config.get_model_path("direct") / model_type
        self.feature_selector = FeatureSelector()

    def evaluate(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        if hours is None:
            hours = list(range(24))

        rows = []
        predictions = {}
        actuals = {}
        dates = None

        for hour in hours:
            model_path = self.model_dir / f"model_H{hour:02d}.pkl"
            if not model_path.exists():
                logger.warning("跳过 H%02d，模型不存在: %s", hour, model_path)
                continue

            data = self._prepare_data(hour)
            model = joblib.load(model_path)
            y_pred = self._apply_calibration(model.predict(data["X_test"]), self._load_calibration(hour))

            rows.append(
                {
                    "hour": hour,
                    "mae": calculate_mae(data["y_test"], y_pred),
                    "rmse": calculate_rmse(data["y_test"], y_pred),
                    "smape": calculate_smape(data["y_test"], y_pred),
                    "n_test": len(data["y_test"]),
                }
            )
            predictions[hour] = y_pred
            actuals[hour] = data["y_test"]
            if dates is None:
                dates = data["test_dates"]

        results_df = pd.DataFrame(rows)
        if results_df.empty:
            raise FileNotFoundError(f"未找到可评估的 Direct 模型: {self.model_dir}")

        self._save_outputs(results_df, predictions, actuals, dates)
        return results_df

    def _prepare_data(self, hour: int) -> Dict[str, Any]:
        engineer = DirectFeatureEngineer()
        features_df, target_col = engineer.load_features(hour)
        feature_cols = self._load_feature_cols(hour, features_df, target_col)

        train_end = int(len(features_df) * self.config.split_config["train_ratio"])
        test_df = features_df.iloc[train_end:].reset_index(drop=True)
        X_test = test_df[feature_cols]

        scaler_path = self.model_dir / f"scaler_H{hour:02d}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        return {
            "X_test": X_test,
            "y_test": test_df[target_col].to_numpy(),
            "test_dates": test_df["预测日期"],
        }

    def _load_feature_cols(self, hour: int, features_df: pd.DataFrame, target_col: str) -> List[str]:
        metadata_path = self.model_dir / f"metadata_H{hour:02d}.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)["feature_cols"]

        candidate_features = [col for col in features_df.columns if col not in [target_col, "预测日期"]]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        return self.feature_selector.select_features_for_model(self.model_type, numeric_features)

    def _load_calibration(self, hour: int) -> Dict[str, float]:
        metadata_path = self.model_dir / f"metadata_H{hour:02d}.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f).get("calibration", {})
        return {}

    @staticmethod
    def _apply_calibration(y_pred: np.ndarray, calibration: Dict[str, float]) -> np.ndarray:
        if not calibration:
            return y_pred
        calibrated = np.asarray(y_pred, dtype=float) * calibration.get("scale", 1.0)
        calibrated = calibrated + calibration.get("bias", 0.0)
        clip_min = calibration.get("clip_min", None)
        if clip_min is not None:
            calibrated = np.maximum(calibrated, clip_min)
        return calibrated

    def _save_outputs(
        self,
        results_df: pd.DataFrame,
        predictions: Dict[int, np.ndarray],
        actuals: Dict[int, np.ndarray],
        dates: pd.Series,
    ) -> None:
        log_dir = self.config.get_result_path("logs") / "direct" / self.model_type
        pred_dir = self.config.get_result_path("predictions") / "direct" / self.model_type
        log_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(log_dir / "evaluation_report.csv", index=False, encoding="utf-8-sig")
        results_df.to_json(log_dir / "evaluation_report.json", orient="records", force_ascii=False, indent=2)

        pred_df = pd.DataFrame({"预测日期": dates})
        for hour in sorted(predictions):
            pred_df[f"actual_H{hour:02d}"] = actuals[hour]
            pred_df[f"pred_H{hour:02d}"] = predictions[hour]
        pred_df.to_csv(pred_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

        summary = {
            "model_type": self.model_type,
            "overall_mae": float(results_df["mae"].mean()),
            "overall_rmse": float(results_df["rmse"].mean()),
            "overall_smape": float(results_df["smape"].mean()),
        }
        with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "evaluate_direct.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct 多步模型评估")
    parser.add_argument("--model", default="lightgbm", choices=list_model_types(), help="基模型类型")
    parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定评估小时")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    evaluator = DirectEvaluator(Config(), args.model)
    results = evaluator.evaluate(args.hours)
    print(results.to_string(index=False))
    print(
        f"\nAverage MAE={results['mae'].mean():.4f}, "
        f"RMSE={results['rmse'].mean():.4f}, "
        f"sMAPE={results['smape'].mean():.2f}%"
    )


if __name__ == "__main__":
    main()
