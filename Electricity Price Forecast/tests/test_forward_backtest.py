import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_split import split_by_months
from probe_optimizer import LightGBMProbeOptimizer
from config import Config
from model_factory import get_default_params, list_model_types


class ForwardBacktestTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "预测日期": pd.date_range("2024-06-12", "2025-06-30", freq="D"),
                "value": 1.0,
            }
        )

    def test_forward_month_splits_do_not_leak_future_data(self):
        expected_train_end = {
            "2025-03": "2025-02-28",
            "2025-04": "2025-03-31",
            "2025-05": "2025-04-30",
            "2025-06": "2025-05-31",
        }

        for month, train_end in expected_train_end.items():
            with self.subTest(month=month):
                split = split_by_months(self.df, "预测日期", [month])
                self.assertEqual(split.train_end, train_end)
                self.assertLess(pd.Timestamp(split.train_end), pd.Timestamp(split.test_start))

    def test_v4_model_alias_is_not_registered(self):
        self.assertNotIn("lightgbm_smape_probe_v4", list_model_types())
        with self.assertRaises(ValueError):
            get_default_params("lightgbm_smape_probe_v4", hour=12)

    def test_probe_optimizer_validates_inside_training_window(self):
        optimizer = LightGBMProbeOptimizer(
            Config(),
            "lightgbm_smape_probe_v3",
            test_months=["2025-04"],
            max_candidates=1,
            cv_folds=3,
            local_alpha_radius=0.1,
            local_alpha_step=0.02,
            broad_alpha_step=0.05,
        )
        df = pd.DataFrame(
            {
                "预测日期": pd.date_range("2025-01-01", "2025-04-30", freq="D"),
                "月份": 1,
                "星期": 1,
                "是否周末": 0,
                "季度": 1,
                "target": 100.0,
            }
        )

        data = optimizer._prepare_data_from_frame(df, "target", 0, ["direct_time"], "2025-04")

        self.assertEqual(data["target_month"], "2025-04")
        self.assertEqual(data["validation_months"], ["2025-02", "2025-03"])
        self.assertEqual(data["split_info"]["cv_folds"], 2)
        self.assertEqual(data["split_info"]["folds"][0]["fit_end"], "2025-01-31")
        self.assertEqual(data["split_info"]["folds"][0]["validation_start"], "2025-02-01")
        self.assertEqual(data["split_info"]["folds"][1]["fit_end"], "2025-02-28")
        self.assertEqual(data["split_info"]["folds"][1]["validation_start"], "2025-03-01")
        self.assertEqual(data["split_info"]["test_start"], "2025-04-01")


if __name__ == "__main__":
    unittest.main()
