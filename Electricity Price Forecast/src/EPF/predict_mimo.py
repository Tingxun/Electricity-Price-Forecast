"""Predict one 24-hour curve with a saved MIMO model."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .evaluate_mimo import MimoEvaluator
from .feature_engineering_mimo import MimoFeatureEngineer
from .train_mimo import MimoTrainer, SUPPORTED_MIMO_MODELS


logger = logging.getLogger(__name__)


class MimoPredictor:
    def __init__(self, config: Config, model_type: str = "tcn_mimo", test_months: Optional[List[str]] = None):
        self.config = config
        self.model_type = model_type
        self.test_months = test_months
        self.evaluator = MimoEvaluator(config, model_type, test_months)

    def predict(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        if target_date is None:
            target_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        metadata, model = self.evaluator._load_model()
        samples = MimoFeatureEngineer(lookback_days=metadata["model_config"]["lookback_days"]).load_features()
        dates = pd.to_datetime(pd.Series(samples["dates"]))
        target_ts = pd.Timestamp(target_date)
        matches = np.where(dates.to_numpy() == target_ts.to_datetime64())[0]
        if len(matches) == 0:
            raise FileNotFoundError(f"MIMO features for target date {target_date} do not exist")
        idx = matches[:1]
        pred = self.evaluator._predict(model, samples, idx, metadata)[0]
        result = {
            "status": "success",
            "strategy": "mimo",
            "model_type": self.model_type,
            "model_dir": str(self.evaluator.model_dir),
            "target_date": target_date,
            "prediction_time": datetime.now().isoformat(timespec="seconds"),
            "predictions": {"hours": list(range(24)), "prices": [float(x) for x in pred]},
            "statistics": {
                "min_price": float(np.min(pred)),
                "max_price": float(np.max(pred)),
                "mean_price": float(np.mean(pred)),
                "std_price": float(np.std(pred)),
            },
        }
        self._save_prediction(result)
        return result

    def _save_prediction(self, result: Dict[str, Any]) -> None:
        run_label = self.evaluator.model_dir.name
        pred_dir = self.config.get_result_path("predictions") / "mimo" / self.model_type / run_label
        pred_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{result['target_date']}_{timestamp}"
        with open(pred_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        pd.DataFrame({"hour": result["predictions"]["hours"], "predicted_price": result["predictions"]["prices"]}).to_csv(
            pred_dir / f"{stem}.csv", index=False, encoding="utf-8-sig"
        )


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with MIMO models")
    parser.add_argument("--model", default="tcn_mimo", choices=SUPPORTED_MIMO_MODELS)
    parser.add_argument("--date", default=None)
    parser.add_argument("--test-months", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    result = MimoPredictor(Config(), args.model, args.test_months).predict(args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
