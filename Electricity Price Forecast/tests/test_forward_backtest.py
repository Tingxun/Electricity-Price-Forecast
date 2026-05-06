import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_split import split_by_months
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

    def test_v4_model_alias_is_registered(self):
        self.assertIn("lightgbm_smape_probe_v4", list_model_types())
        self.assertIsInstance(get_default_params("lightgbm_smape_probe_v4", hour=12), dict)


if __name__ == "__main__":
    unittest.main()
