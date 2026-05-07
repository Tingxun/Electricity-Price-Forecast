import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.auto_model_selection import generate_auto_candidates, monthly_time_series_folds
from utils.data_split import split_by_months
from model_factory import get_default_params, list_model_types
from utils.model_store import safe_period_label


class ForwardBacktestTests(unittest.TestCase):
    def setUp(self):
        self.date_col = "预测日期"
        self.df = pd.DataFrame(
            {
                self.date_col: pd.date_range("2024-06-12", "2025-06-30", freq="D"),
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
                split = split_by_months(self.df, self.date_col, [month])
                self.assertEqual(split.train_end, train_end)
                self.assertLess(pd.Timestamp(split.train_end), pd.Timestamp(split.test_start))

    def test_v3_v4_model_aliases_are_not_registered(self):
        self.assertIn("lightgbm_auto", list_model_types())
        self.assertNotIn("lightgbm_smape_probe_v3", list_model_types())
        self.assertNotIn("lightgbm_smape_probe_v4", list_model_types())
        with self.assertRaises(ValueError):
            get_default_params("lightgbm_smape_probe_v3", hour=12)
        with self.assertRaises(ValueError):
            get_default_params("lightgbm_smape_probe_v4", hour=12)

    def test_model_run_period_label_is_filesystem_safe(self):
        self.assertEqual(safe_period_label("2025-03"), "2025-03")
        self.assertEqual(safe_period_label("2025-03,2025-04"), "2025-03__2025-04")

    def test_monthly_time_series_cv_uses_only_past_months(self):
        folds = monthly_time_series_folds(self.df[self.date_col], cv_folds=3)
        months = pd.to_datetime(self.df[self.date_col]).dt.to_period("M")
        for train_idx, val_idx, validation_month in folds:
            with self.subTest(validation_month=validation_month):
                self.assertTrue((months.iloc[train_idx] < pd.Period(validation_month, freq="M")).all())
                self.assertTrue((months.iloc[val_idx] == pd.Period(validation_month, freq="M")).all())

    def test_auto_candidates_gate_midday_and_low_price_structures(self):
        default_params = {"objective": "regression_l1", "n_estimators": 10, "random_state": 42}
        high_prices = [100.0] * len(self.df)
        midday_candidates = generate_auto_candidates(12, default_params, high_prices, cv_folds=None)
        night_candidates = generate_auto_candidates(2, default_params, high_prices, cv_folds=None)

        self.assertTrue(any("direct_midday_regime" in c.feature_groups for c in midday_candidates))
        self.assertFalse(any("direct_midday_regime" in c.feature_groups for c in night_candidates))
        self.assertFalse(any(c.structure == "two_stage_low_price" for c in midday_candidates))


if __name__ == "__main__":
    unittest.main()
