import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from EPF.feature_engineering.mimo import HOUR_COL, PRICE_COL, MimoFeatureEngineer
from EPF.models.tcn_mimo import TcnMimoConfig, TcnMimoNet
from EPF.strategies.mimo.train import MimoTrainer
from EPF.utils.strategy_registry import implemented_strategy_names


class MimoWorkflowTests(unittest.TestCase):
    def _frame(self, n_days: int = 24) -> pd.DataFrame:
        rows = []
        base = pd.Timestamp("2025-01-01")
        for day in range(n_days):
            date = base + pd.Timedelta(days=day)
            for hour in range(24):
                rows.append(
                    {
                        "日期": date,
                        "时段": f"{hour + 1:02d}:00",
                        HOUR_COL: hour,
                        "系统负荷-实时": 1000 + day * 10 + hour,
                        "气象-温度-预测": 20 + hour,
                        "气象-温度-实际": 99 + hour,
                        PRICE_COL: day * 100 + hour,
                    }
                )
        return pd.DataFrame(rows)

    def test_mimo_features_use_d_minus_2_and_exclude_leaky_columns(self):
        samples = MimoFeatureEngineer().create_features(self._frame())

        self.assertEqual(samples["price_history"].shape[1:], (14, 24))
        self.assertEqual(samples["target_exog"].shape[1], 24)
        self.assertEqual(samples["targets"].shape[1], 24)
        self.assertNotIn(PRICE_COL, samples["exog_columns"])
        self.assertFalse(any("实际" in col for col in samples["exog_columns"]))

        first_date = pd.Timestamp(str(samples["dates"][0]))
        history_dates = [pd.Timestamp(str(x)) for x in samples["history_dates"][0]]
        self.assertEqual(history_dates[-1], first_date - pd.Timedelta(days=2))
        self.assertEqual(history_dates[0], first_date - pd.Timedelta(days=15))
        self.assertEqual(float(samples["price_history"][0, -1, 0]), 1300.0)

    def test_tcn_mimo_output_shape(self):
        model = TcnMimoNet(exog_dim=3, config=TcnMimoConfig(hidden_channels=8, tcn_levels=1))
        pred = model(
            price_history=torch.zeros((2, 14, 24), dtype=torch.float32),
            target_exog=torch.zeros((2, 24, 3), dtype=torch.float32),
        )
        self.assertEqual(tuple(pred.shape), (2, 24))

    def test_mimo_strategy_is_implemented(self):
        self.assertIn("mimo", implemented_strategy_names())

    def test_mimo_metadata_contains_required_training_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engineer = MimoFeatureEngineer()
            samples = engineer.create_features(self._frame(80))
            feature_dir = Path(tmpdir) / "features"
            engineer.save_features(samples, feature_dir)
            loaded = engineer.load_features(feature_dir)
            split = MimoTrainer._month_split(loaded["dates"], ["2025-03"])
            train_idx, val_idx = MimoTrainer._train_val_split(loaded["dates"], split["train_idx"])
            trainer = MimoTrainer.__new__(MimoTrainer)
            scalers = MimoTrainer._fit_scalers(trainer, loaded, train_idx)

            self.assertIn("price_mean", scalers)
            self.assertIn("target_std", scalers)
            self.assertIn("clip_min", scalers)
            self.assertIn("clip_max", scalers)
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
            self.assertGreater(len(split["test_idx"]), 0)


if __name__ == "__main__":
    unittest.main()
