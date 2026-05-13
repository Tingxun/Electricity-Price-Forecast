"""Feature engineering for true MIMO 24-hour curve forecasting."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config


logger = logging.getLogger(__name__)

DATE_COL = "\u65e5\u671f"
HOUR_COL = "\u5c0f\u65f6"
PERIOD_COL = "\u65f6\u6bb5"
TARGET_DATE_COL = "\u9884\u6d4b\u65e5\u671f"
PRICE_COL = "\u5e73\u5747\u51fa\u6e05\u4ef7\u683c-\u5b9e\u65f6\uff08\u5143/MWh\uff09"


class MimoFeatureEngineer:
    """Build one daily sample that predicts the full 24-hour price curve."""

    def __init__(self, lookback_days: int = 14, price_lag_days: int = 2):
        self.lookback_days = lookback_days
        self.price_lag_days = price_lag_days

    def create_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self._normalize_input(df)
        exog_cols = self._select_exog_columns(df)
        daily_data = {date: group.sort_values(HOUR_COL).reset_index(drop=True) for date, group in df.groupby(DATE_COL)}
        all_dates = sorted(daily_data)

        dates: List[str] = []
        price_history: List[np.ndarray] = []
        target_exog: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        used_history_dates: List[List[str]] = []

        min_index = self.lookback_days + self.price_lag_days - 1
        for idx in range(min_index, len(all_dates)):
            target_date = all_dates[idx]
            target_day = daily_data[target_date]
            if not self._is_complete_day(target_day):
                continue

            history_start = idx - self.lookback_days - self.price_lag_days + 1
            history_end = idx - self.price_lag_days + 1
            history_dates = all_dates[history_start:history_end]
            if len(history_dates) != self.lookback_days:
                continue
            history_days = [daily_data[date] for date in history_dates]
            if not all(self._is_complete_day(day) for day in history_days):
                continue

            hist_prices = np.stack([day[PRICE_COL].to_numpy(dtype=np.float32) for day in history_days], axis=0)
            exog = target_day[exog_cols].to_numpy(dtype=np.float32)
            y = target_day[PRICE_COL].to_numpy(dtype=np.float32)
            if np.isnan(hist_prices).any() or np.isnan(exog).any() or np.isnan(y).any():
                continue

            dates.append(pd.Timestamp(target_date).strftime("%Y-%m-%d"))
            price_history.append(hist_prices)
            target_exog.append(exog)
            targets.append(y)
            used_history_dates.append([pd.Timestamp(date).strftime("%Y-%m-%d") for date in history_dates])

        if not dates:
            raise ValueError("no MIMO samples could be generated")

        return {
            "dates": np.asarray(dates),
            "price_history": np.stack(price_history, axis=0),
            "target_exog": np.stack(target_exog, axis=0),
            "targets": np.stack(targets, axis=0),
            "history_dates": np.asarray(used_history_dates),
            "exog_columns": exog_cols,
        }

    def save_features(self, samples: Dict[str, Any], feature_path: Optional[Path] = None) -> None:
        config = Config()
        if feature_path is None:
            feature_path = config.get_data_path("features") / "mimo"
        feature_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            feature_path / "mimo_samples.npz",
            dates=samples["dates"],
            price_history=samples["price_history"],
            target_exog=samples["target_exog"],
            targets=samples["targets"],
            history_dates=samples["history_dates"],
            exog_columns=np.asarray(samples["exog_columns"]),
        )
        info = {
            "type": "mimo_daily_curve",
            "n_samples": int(len(samples["dates"])),
            "lookback_days": self.lookback_days,
            "price_lag_days": self.price_lag_days,
            "price_history_shape": list(samples["price_history"].shape[1:]),
            "target_exog_shape": list(samples["target_exog"].shape[1:]),
            "target_shape": list(samples["targets"].shape[1:]),
            "date_col": DATE_COL,
            "hour_col": HOUR_COL,
            "price_col": PRICE_COL,
            "exog_columns": list(samples["exog_columns"]),
            "leakage_rules": {
                "target_day_price_excluded": True,
                "history_price_available_until": "D-2",
                "exclude_actual_weather_columns": True,
                "target_day_exog_allowed": True,
            },
        }
        with open(feature_path / "feature_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info("MIMO features saved to %s", feature_path)

    def load_features(self, feature_path: Optional[Path] = None) -> Dict[str, Any]:
        config = Config()
        if feature_path is None:
            feature_path = config.get_data_path("features") / "mimo"
        data = np.load(feature_path / "mimo_samples.npz", allow_pickle=True)
        return {
            "dates": data["dates"].astype(str),
            "price_history": data["price_history"].astype(np.float32),
            "target_exog": data["target_exog"].astype(np.float32),
            "targets": data["targets"].astype(np.float32),
            "history_dates": data["history_dates"].astype(str),
            "exog_columns": data["exog_columns"].astype(str).tolist(),
        }

    def _normalize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = [col for col in [DATE_COL, HOUR_COL, PRICE_COL] if col not in df.columns]
        if missing:
            raise ValueError(f"processed data missing required columns: {missing}")
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df[HOUR_COL] = pd.to_numeric(df[HOUR_COL], errors="coerce").astype("Int64")
        df = df.sort_values([DATE_COL, HOUR_COL]).reset_index(drop=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def _select_exog_columns(self, df: pd.DataFrame) -> List[str]:
        excluded = {DATE_COL, PERIOD_COL, PRICE_COL}
        result = []
        for col in df.columns:
            if col in excluded:
                continue
            if "\u5b9e\u9645" in col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                result.append(col)
        if PRICE_COL in result:
            raise ValueError("target price leaked into MIMO exogenous features")
        if not result:
            raise ValueError("no numeric exogenous features selected for MIMO")
        return result

    @staticmethod
    def _is_complete_day(day: pd.DataFrame) -> bool:
        if len(day) != 24:
            return False
        hours = set(pd.to_numeric(day[HOUR_COL], errors="coerce").dropna().astype(int).tolist())
        return hours == set(range(24))


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIMO daily curve features")
    parser.add_argument("--lookback-days", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = Config()
    processed_file = config.get_data_path("processed_data")
    df = pd.read_csv(processed_file)
    engineer = MimoFeatureEngineer(lookback_days=args.lookback_days)
    samples = engineer.create_features(df)
    engineer.save_features(samples)


if __name__ == "__main__":
    main()
