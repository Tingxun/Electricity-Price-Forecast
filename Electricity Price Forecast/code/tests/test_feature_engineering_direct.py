import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from feature_engineering_direct import DirectFeatureEngineer


class DirectFeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.engineer = DirectFeatureEngineer()

    def _build_synthetic_frame(self, n_days: int = 20) -> pd.DataFrame:
        rows = []
        base_date = pd.Timestamp("2025-01-01")
        for day_offset in range(n_days):
            date = base_date + pd.Timedelta(days=day_offset)
            for hour in range(24):
                row = {
                    "日期": date,
                    "小时": hour,
                    self.engineer.price_col: 100.0 + day_offset + hour,
                    "温度-预测": 10.0 + hour,
                    "总云量-预测": 20.0 + hour,
                    "辐照度-预测": 30.0 + hour,
                }
                for idx, col in enumerate(self.engineer.market_cols):
                    row[col] = 1000.0 + idx * 100 + day_offset * 10 + hour
                rows.append(row)
        return pd.DataFrame(rows)

    def test_build_direct_features_uses_second_lag_instead_of_future_hour(self):
        df = self._build_synthetic_frame()
        df = df.sort_values(["日期", "小时"]).reset_index(drop=True)
        all_dates = df["日期"].unique()
        daily_data = {date: group.copy() for date, group in df.groupby("日期", sort=False)}

        date_index = 15
        target_hour = 10
        features = self.engineer._build_direct_features(
            daily_data=daily_data,
            all_dates=all_dates,
            date_index=date_index,
            t_minus_2_data=daily_data[all_dates[date_index - 2]],
            target_date_data=daily_data[all_dates[date_index]],
            target_date=all_dates[date_index],
            target_hour=target_hour,
        )

        self.assertIsNotNone(features)
        self.assertTrue(any(key.startswith("滞后2h_市场_") for key in features))
        self.assertTrue(any(key.startswith("滞后6h_市场_") for key in features))
        self.assertTrue(any(key.startswith("滞后3h_气象_") for key in features))
        self.assertFalse(any(key.startswith("当前_气象_") for key in features))
        self.assertFalse(any(key.startswith("滞后1h_气象_") for key in features))
        self.assertFalse(any(key.startswith("滞后2h_气象_") for key in features))
        self.assertTrue(any(key.startswith("滞后2h_气象聚合_") for key in features))
        self.assertTrue(any(key.startswith("市场变化_滞后1h减滞后2h_") for key in features))
        self.assertTrue(any(key.startswith("市场变化_当前减滞后6h_") for key in features))
        self.assertFalse(any("未来1h" in key for key in features))
        self.assertNotIn("月份", features)
        self.assertNotIn("星期", features)
        self.assertNotIn("是否周末", features)
        self.assertNotIn("季度", features)
        self.assertIn("星期_3", features)
        self.assertFalse(any(key.startswith("月份_") for key in features))
        self.assertFalse(any("低价次数" in key for key in features))
        self.assertFalse(any("近零次数" in key for key in features))
        self.assertFalse(any("零价次数" in key for key in features))
        self.assertFalse(any("占负荷比" in key for key in features))
        self.assertFalse(any(key.endswith("_最大值") and "气象聚合" in key for key in features))


if __name__ == "__main__":
    unittest.main()
