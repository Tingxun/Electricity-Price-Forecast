"""Create robust comparison reports across Direct and MIMO backtests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config


def _load_overall(path: Path, strategy: str, model_type: str, retrain_frequency: str) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "strategy": strategy,
        "model_type": model_type,
        "retrain_frequency": retrain_frequency,
        "overall_smape": data.get("overall_smape"),
        "overall_mae": data.get("overall_mae"),
        "overall_rmse": data.get("overall_rmse"),
        "overall_acc_rate": data.get("overall_acc_rate"),
        "robust_score": data.get("robust_score"),
        "worst_month_smape": data.get("worst_month_smape"),
        "midday_smape": data.get("midday_smape"),
        "non_midday_smape": data.get("non_midday_smape"),
        "meets_overall_lt_40": (data.get("overall_smape") or 999.0) < 40.0,
        "meets_worst_month_lt_55": (data.get("worst_month_smape") or 999.0) < 55.0,
        "source": str(path),
    }


def build_comparison(config: Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    specs = [
        ("direct", "lightgbm_auto", "monthly", config.get_result_path("logs") / "direct" / "lightgbm_auto" / "monthly_backtest_overall.json"),
        ("direct", "lightgbm_auto", "weekly", config.get_result_path("logs") / "direct" / "lightgbm_auto" / "weekly_backtest_overall.json"),
        ("mimo", "tcn_mimo", "monthly", config.get_result_path("logs") / "mimo" / "tcn_mimo" / "monthly_backtest_overall.json"),
        ("mimo", "tcn_mimo", "weekly", config.get_result_path("logs") / "mimo" / "tcn_mimo" / "weekly_backtest_overall.json"),
    ]
    for strategy, model_type, frequency, path in specs:
        row = _load_overall(path, strategy, model_type, frequency)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["overall_smape", "robust_score"], na_position="last")
    return df


def save_comparison(config: Config) -> pd.DataFrame:
    df = build_comparison(config)
    out_dir = config.get_result_path("logs") / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "robust_comparison.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "robust_comparison.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare robust backtest reports")
    parser.parse_args()
    df = save_comparison(Config())
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
