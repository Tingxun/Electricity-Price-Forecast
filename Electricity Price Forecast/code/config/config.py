"""
Project configuration for the Direct multi-step forecasting workflow.
"""

from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Centralized paths and experiment settings."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.data_paths = {
            "processed_data": self.project_root / "data" / "processed" / "processed_data.csv",
            "features": self.project_root / "data" / "features",
            "direct_features": self.project_root / "data" / "features" / "direct",
            "direct_feature_info": self.project_root / "data" / "features" / "direct" / "feature_info.json",
        }

        self.model_paths = {
            "direct": self.project_root / "saved_models" / "direct",
        }

        self.result_paths = {
            "predictions": self.project_root / "results" / "predictions",
            "figures": self.project_root / "results" / "figures",
            "logs": self.project_root / "results" / "logs",
        }

        self.data_config = {
            "start_date": "2024-05-28",
            "end_date": "2025-03-26",
            "hours_per_day": 24,
            "forecast_horizon": 24,
        }

        self.split_config = {
            "test_months": None,
            "cv_folds": 3,
            "random_seed": 42,
            "rolling_min_train_months": 3,
        }

        self.training_config = {
            "default_model": "lightgbm",
            "default_n_iter": 20,
        }

        self._create_directories()

    def _create_directories(self) -> None:
        for path in [*self.data_paths.values(), *self.model_paths.values(), *self.result_paths.values()]:
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

    def get_data_path(self, key: str) -> Optional[Path]:
        return self.data_paths.get(key)

    def get_model_path(self, model_type: str = "direct") -> Optional[Path]:
        return self.model_paths.get(model_type)

    def get_result_path(self, result_type: str) -> Optional[Path]:
        return self.result_paths.get(result_type)

    def get_split_config(self) -> Dict[str, Any]:
        return self.split_config.copy()


config = Config()
