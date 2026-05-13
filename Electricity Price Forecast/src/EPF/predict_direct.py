"""
Predict a 24-hour price curve with trained Direct models.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .feature_engineering_direct import DirectFeatureEngineer
from .feature_selector import FeatureSelector
from .utils.model_store import resolve_model_dir
from .model_factory import list_model_types


logger = logging.getLogger(__name__)


class DirectPredictor:
    """Load 24 hourly models and predict one target date."""

    def __init__(self, config: Config, model_type: str, test_months: Optional[List[str]] = None):
        self.config = config
        self.model_type = model_type
        self.test_months = test_months
        self.model_dir = resolve_model_dir(config, model_type, test_months)
        self.feature_selector = FeatureSelector()

    def predict(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        if target_date is None:
            target_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        prices = []
        used_date = None
        for hour in range(24):
            price, row_date = self._predict_hour(hour, target_date)
            prices.append(float(price))
            used_date = row_date

        result = {
            "status": "success",
            "model_type": self.model_type,
            "model_dir": str(self.model_dir),
            "target_date": target_date,
            "feature_date_used": str(used_date.date()) if hasattr(used_date, "date") else str(used_date),
            "prediction_time": datetime.now().isoformat(timespec="seconds"),
            "predictions": {
                "hours": list(range(24)),
                "prices": prices,
            },
            "statistics": {
                "min_price": float(np.min(prices)),
                "max_price": float(np.max(prices)),
                "mean_price": float(np.mean(prices)),
                "std_price": float(np.std(prices)),
            },
        }
        self._save_prediction(result)
        return result

    def _predict_hour(self, hour: int, target_date: str) -> Tuple[float, pd.Timestamp]:
        model_path = self.model_dir / f"model_H{hour:02d}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"缺少 H{hour:02d} 模型: {model_path}")

        engineer = DirectFeatureEngineer()
        features_df, target_col = engineer.load_features(hour)
        features_df["预测日期"] = pd.to_datetime(features_df["预测日期"])

        target_ts = pd.to_datetime(target_date)
        row = features_df[features_df["预测日期"] == target_ts]
        if row.empty:
            row = features_df.tail(1)
            logger.warning("H%02d 未找到目标日期 %s，使用最新特征日期 %s", hour, target_date, row.iloc[0]["预测日期"])

        feature_cols = self._load_feature_cols(hour, features_df, target_col)
        X = row[feature_cols]

        scaler_path = self.model_dir / f"scaler_H{hour:02d}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X = pd.DataFrame(scaler.transform(X), columns=feature_cols)

        model = joblib.load(model_path)
        pred = self._apply_calibration(model.predict(X), self._load_calibration(hour))
        return float(np.asarray(pred).reshape(-1)[0]), row.iloc[0]["预测日期"]

    def _load_feature_cols(self, hour: int, features_df: pd.DataFrame, target_col: str) -> List[str]:
        metadata_path = self.model_dir / f"metadata_H{hour:02d}.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)["feature_cols"]

        candidate_features = [col for col in features_df.columns if col not in [target_col, "预测日期"]]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        return self.feature_selector.select_features_for_model(self.model_type, numeric_features, hour=hour)

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

    def _save_prediction(self, result: Dict[str, Any]) -> None:
        run_label = self.model_dir.name if self.model_dir.parent.name == self.model_type else "legacy"
        pred_dir = self.config.get_result_path("predictions") / "direct" / self.model_type / run_label
        pred_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{result['target_date']}_{timestamp}"

        with open(pred_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        pd.DataFrame(
            {
                "hour": result["predictions"]["hours"],
                "predicted_price": result["predictions"]["prices"],
            }
        ).to_csv(pred_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "predict_direct.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct 多步模型预测")
    parser.add_argument("--model", default="lightgbm", choices=list_model_types(), help="基模型类型")
    parser.add_argument("--date", default=None, help="预测目标日期 YYYY-MM-DD")
    parser.add_argument("--test-months", nargs="+", default=None, help="选择用哪个测试月份训练出的模型；默认使用最新训练版本")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    result = DirectPredictor(Config(), args.model, test_months=args.test_months).predict(args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
