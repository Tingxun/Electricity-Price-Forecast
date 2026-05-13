"""Shared column names and labels used across EPF workflows."""

RAW_DATE_COL = "日期"
RAW_PERIOD_COL = "时段"
HOUR_COL = "小时"
PRED_DATE_COL = "预测日期"
PRICE_COL = "平均出清价格-实时（元/MWh）"
ACTUAL_MARKER = "实际"

ROLLING_MODE_MONTHLY = "expanding_forward"
ROLLING_MODE_WEEKLY = "expanding_forward_weekly"

MIDDAY_HOURS = tuple(range(8, 16))
NON_MIDDAY_HOURS = tuple([*range(0, 8), *range(16, 24)])
