"""Month-based data splitting helpers for Direct forecasting."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class MonthSplit:
    train_mask: pd.Series
    test_mask: pd.Series
    test_months: List[str]
    test_period: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int

    def to_dict(self) -> dict:
        return {
            "split_strategy": "month",
            "test_months": self.test_months,
            "test_period": self.test_period,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def split_by_months(
    df: pd.DataFrame,
    date_col: str,
    test_months: Optional[Sequence[str] | str] = None,
) -> MonthSplit:
    """Return masks for a one-month or multi-month holdout split.

    Parameters
    ----------
    df:
        Feature dataframe sorted by date.
    date_col:
        Date column name.
    test_months:
        Month label(s) in ``YYYY-MM`` format. If omitted, use the last
        available month in ``df``.
    """
    dates = pd.to_datetime(df[date_col])
    months = dates.dt.to_period("M")
    unique_months = sorted(months.unique())
    if len(unique_months) < 2:
        raise ValueError("至少需要两个月数据才能按月份划分训练集和测试集")

    selected = _parse_test_months(test_months, unique_months)
    missing = [month for month in selected if month not in unique_months]
    if missing:
        available = ", ".join(str(month) for month in unique_months)
        requested = ", ".join(str(month) for month in missing)
        raise ValueError(f"测试月份 {requested} 不在特征数据中。可选月份: {available}")

    first_test_month = min(selected)
    train_mask = dates < first_test_month.start_time
    test_mask = months.isin(selected)

    if not train_mask.any():
        raise ValueError(f"测试月份 {first_test_month} 之前没有训练数据")
    if not test_mask.any():
        raise ValueError(f"测试月份 {format_test_months(selected)} 没有测试样本")

    train_dates = dates[train_mask]
    test_dates = dates[test_mask]
    test_month_labels = [str(month) for month in selected]
    return MonthSplit(
        train_mask=train_mask,
        test_mask=test_mask,
        test_months=test_month_labels,
        test_period=format_test_months(selected),
        train_start=_fmt_date(train_dates.min()),
        train_end=_fmt_date(train_dates.max()),
        test_start=_fmt_date(test_dates.min()),
        test_end=_fmt_date(test_dates.max()),
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
    )


def format_test_months(test_months: Sequence[pd.Period]) -> str:
    ordered = sorted(test_months)
    return ",".join(str(month) for month in ordered)


def list_rolling_months(
    df: pd.DataFrame,
    date_col: str,
    min_train_months: int = 3,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> List[str]:
    """List valid test months for expanding-window monthly backtesting."""
    dates = pd.to_datetime(df[date_col])
    months = sorted(dates.dt.to_period("M").unique())
    start_period = pd.Period(start_month, freq="M") if start_month else None
    end_period = pd.Period(end_month, freq="M") if end_month else None

    result = []
    for idx, month in enumerate(months):
        if idx < min_train_months:
            continue
        if start_period and month < start_period:
            continue
        if end_period and month > end_period:
            continue
        result.append(str(month))
    return result


def _fmt_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _parse_test_months(
    test_months: Optional[Sequence[str] | str],
    available_months: Sequence[pd.Period],
) -> List[pd.Period]:
    if test_months is None:
        return [available_months[-1]]
    if isinstance(test_months, str):
        raw_months = [item.strip() for item in test_months.split(",") if item.strip()]
    else:
        raw_months = []
        for item in test_months:
            raw_months.extend(part.strip() for part in str(item).split(",") if part.strip())

    if not raw_months:
        return [available_months[-1]]
    return sorted({pd.Period(month, freq="M") for month in raw_months})
