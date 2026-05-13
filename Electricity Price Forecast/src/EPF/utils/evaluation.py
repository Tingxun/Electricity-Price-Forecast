"""Shared evaluation helpers for forecasting reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from utils.metrics import calculate_sape


def price_bucket(value: float) -> str:
    value = float(value)
    if value <= 0:
        return "zero"
    if value <= 20:
        return "near_zero"
    if value <= 80:
        return "low_20_80"
    if value <= 200:
        return "mid_80_200"
    return "high_200_plus"


def prediction_rows_from_wide(
    dates: Iterable[Any],
    actual: np.ndarray,
    pred: np.ndarray,
    *,
    test_months: Optional[Iterable[str]] = None,
    rolling_mode: Optional[str] = None,
    week_id: Optional[str] = None,
    week_start: Optional[str] = None,
    week_end: Optional[str] = None,
) -> pd.DataFrame:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError(f"actual and pred shape mismatch: {actual.shape} vs {pred.shape}")
    if actual.ndim != 2 or actual.shape[1] != 24:
        raise ValueError(f"expected [n_samples, 24] arrays, got {actual.shape}")

    date_series = pd.to_datetime(pd.Series(list(dates)))
    month_labels = list(test_months) if test_months is not None else date_series.dt.to_period("M").astype(str).tolist()
    if len(month_labels) != len(date_series):
        month_labels = date_series.dt.to_period("M").astype(str).tolist()

    sape = calculate_sape(actual.reshape(-1), pred.reshape(-1)).reshape(actual.shape)
    rows = []
    for i, date in enumerate(date_series):
        for hour in range(24):
            actual_value = float(actual[i, hour])
            pred_value = float(pred[i, hour])
            rows.append(
                {
                    "rolling_mode": rolling_mode,
                    "test_month": month_labels[i],
                    "week_id": week_id,
                    "week_start": week_start,
                    "week_end": week_end,
                    "预测日期": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "actual": actual_value,
                    "pred": pred_value,
                    "error": pred_value - actual_value,
                    "abs_error": abs(pred_value - actual_value),
                    "sape": float(sape[i, hour]),
                    "actual_price_bucket": price_bucket(actual_value),
                }
            )
    return pd.DataFrame(rows)


def summarize_predictions(prediction_df: pd.DataFrame) -> Dict[str, Any]:
    if prediction_df.empty:
        return {
            "overall": {},
            "month_summary": pd.DataFrame(),
            "hour_summary": pd.DataFrame(),
            "bucket_summary": pd.DataFrame(),
            "robust_score": None,
        }

    df = prediction_df.copy()
    overall = {
        "overall_mae": float(df["abs_error"].mean()),
        "overall_rmse": float(np.sqrt(np.mean(np.square(df["error"])))),
        "overall_smape": float(df["sape"].mean()),
        "overall_acc_rate": float((df["sape"] < 20.0).mean() * 100.0),
    }

    month_summary = (
        df.groupby("test_month", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            smape=("sape", "mean"),
            monthly_acc_rate=("sape", lambda s: float((s < 20.0).mean() * 100.0)),
        )
        .sort_values("test_month")
    )

    hour_summary = (
        df.groupby(["test_month", "hour"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            smape=("sape", "mean"),
            acc_rate=("sape", lambda s: float((s < 20.0).mean() * 100.0)),
        )
        .sort_values(["test_month", "hour"])
    )

    bucket_summary = (
        df.groupby(["test_month", "actual_price_bucket"], as_index=False)
        .agg(
            n=("sape", "size"),
            mae=("abs_error", "mean"),
            smape=("sape", "mean"),
            acc_rate=("sape", lambda s: float((s < 20.0).mean() * 100.0)),
        )
        .sort_values(["test_month", "actual_price_bucket"])
    )

    midday = hour_summary[hour_summary["hour"].between(8, 15)]
    non_midday = hour_summary[~hour_summary["hour"].between(8, 15)]
    midday_smape = float(midday["smape"].mean()) if not midday.empty else overall["overall_smape"]
    non_midday_smape = float(non_midday["smape"].mean()) if not non_midday.empty else overall["overall_smape"]
    month_std = float(month_summary["smape"].std(ddof=0)) if len(month_summary) > 1 else 0.0
    worst_month_smape = float(month_summary["smape"].max())
    robust_score = (
        overall["overall_smape"]
        + 0.30 * month_std
        + 0.20 * max(0.0, worst_month_smape - 45.0)
        + 0.20 * max(0.0, midday_smape - non_midday_smape)
    )

    month_summary["midday_smape"] = month_summary["test_month"].map(_band_smape(hour_summary, range(8, 16)))
    month_summary["non_midday_smape"] = month_summary["test_month"].map(
        _band_smape(hour_summary, [*range(0, 8), *range(16, 24)])
    )
    month_summary["worst_hours"] = month_summary["test_month"].map(_worst_hours(hour_summary))
    month_summary["smape_below_40"] = month_summary["smape"] < 40.0
    month_summary["smape_below_45"] = month_summary["smape"] < 45.0

    overall.update(
        {
            "month_std_smape": month_std,
            "worst_month_smape": worst_month_smape,
            "midday_smape": midday_smape,
            "non_midday_smape": non_midday_smape,
            "robust_score": float(robust_score),
            "months_below_40": int(month_summary["smape_below_40"].sum()),
            "months_below_45": int(month_summary["smape_below_45"].sum()),
        }
    )
    return {
        "overall": overall,
        "month_summary": month_summary,
        "hour_summary": hour_summary,
        "bucket_summary": bucket_summary,
        "robust_score": float(robust_score),
    }


def save_prediction_report(
    prediction_df: pd.DataFrame,
    log_dir: Path,
    prefix: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(log_dir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    summary = summarize_predictions(prediction_df)
    summary["month_summary"].to_csv(log_dir / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    summary["hour_summary"].to_csv(log_dir / f"{prefix}_hour_summary.csv", index=False, encoding="utf-8-sig")
    summary["bucket_summary"].to_csv(log_dir / f"{prefix}_bucket_summary.csv", index=False, encoding="utf-8-sig")
    overall = dict(metadata or {})
    overall.update(summary["overall"])
    with open(log_dir / f"{prefix}_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    return overall


def _band_smape(results_df: pd.DataFrame, hours: Iterable[int]) -> Dict[str, float]:
    hour_set = set(hours)
    band_df = results_df[results_df["hour"].isin(hour_set)]
    return band_df.groupby("test_month")["smape"].mean().to_dict()


def _worst_hours(results_df: pd.DataFrame, top_n: int = 5) -> Dict[str, str]:
    result = {}
    for month, group in results_df.groupby("test_month"):
        worst = group.sort_values("smape", ascending=False).head(top_n)
        result[month] = ";".join(f"H{int(row.hour):02d}:{float(row.smape):.2f}" for row in worst.itertuples())
    return result
