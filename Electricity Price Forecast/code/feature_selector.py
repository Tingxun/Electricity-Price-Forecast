"""Feature selection for Direct multi-step models."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


FEATURE_CONFIG: Dict = {
    "feature_groups": {
        "direct_time": {
            "features": ["月份", "星期", "是否周末", "季度"],
        },
        "direct_time_midday": {
            "features": [
                "月份",
                "星期",
                "是否周末",
                "季度",
                "星期_1",
                "星期_2",
                "星期_3",
                "星期_4",
                "星期_5",
                "星期_6",
                "星期_7",
                "是否周三",
                "是否周六",
                "是否周日",
            ],
        },
        "direct_price_lag": {
            "patterns": [
                r"^滞后\d+天_H\d{2}_价格$",
                r"^滞后2天_H\d{2}_(前\d+h|后\d+h)_价格$",
                r"^历史价格_",
                r"^泛化历史价格_",
            ],
        },
        "direct_market_window": {
            "patterns": [
                r"^(当前|滞后1h|未来1h)_市场_",
                r"^市场变化_",
                r"^市场日形态_",
            ],
        },
        "direct_weather_window": {
            "patterns": [
                r"^(当前|滞后1h|未来1h)_气象_",
            ],
        },
        "direct_midday_regime": {
            "patterns": [
                r"^同星期历史_",
                r"^午间低价_",
                r"^泛化午间低价_",
                r"^午间市场形态_",
            ],
        },
        "direct_midday_weather_agg": {
            "patterns": [
                r"^(当前|滞后1h|未来1h)_气象聚合_",
                r"^午间气象聚合_",
            ],
        },
    },
    "default": {
        "description": "Direct 默认特征集",
        "feature_groups": [
            "direct_time",
            "direct_price_lag",
            "direct_market_window",
            "direct_weather_window",
        ],
        "normalize": False,
    },
    "model_features": {
        "lightgbm": {
            "description": "LightGBM Direct baseline",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window"],
            "normalize": False,
        },
        "lightgbm_auto": {
            "description": "LightGBM Direct 自动结构/特征组选择模型",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window"],
            "normalize": False,
        },
        "lightgbm_smape_probe": {
            "description": "LightGBM Direct sMAPE 固定参数兼容基线",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window"],
            "normalize": False,
        },
        "xgboost": {
            "description": "XGBoost Direct baseline",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window", "direct_weather_window"],
            "normalize": False,
        },
        "random_forest": {
            "description": "RandomForest Direct baseline",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window", "direct_weather_window"],
            "normalize": False,
        },
        "ridge": {
            "description": "Ridge Direct linear baseline",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window", "direct_weather_window"],
            "normalize": True,
        },
        "lasso": {
            "description": "Lasso Direct linear baseline",
            "feature_groups": ["direct_time", "direct_price_lag", "direct_market_window", "direct_weather_window"],
            "normalize": True,
        },
    },
}


class FeatureSelector:
    """Select model-specific features from available columns."""

    def __init__(self, config: Dict | None = None):
        self.config = config or FEATURE_CONFIG

    def get_model_features(self, model_name: str) -> Dict:
        model_features = self.config.get("model_features", {})
        return model_features.get(model_name, self.config.get("default", {}))

    def select_features_for_model(
        self,
        model_name: str,
        available_features: List[str],
        hour: int | None = None,
        feature_groups: List[str] | None = None,
    ) -> List[str]:
        model_config = self.get_model_features(model_name)
        if hour is not None:
            hour_override = model_config.get("hourly_overrides", {}).get(hour)
            if hour_override is None:
                hour_override = model_config.get("hourly_overrides", {}).get(str(hour))
            if hour_override:
                model_config = {**model_config, **hour_override}
        if feature_groups is not None:
            model_config = {**model_config, "feature_groups": feature_groups}

        return self._select_features(model_name, model_config, available_features)

    def select_features_from_groups(self, feature_groups: List[str], available_features: List[str]) -> List[str]:
        return self._select_features("custom", {"feature_groups": feature_groups}, available_features)

    def _select_features(self, model_name: str, model_config: Dict, available_features: List[str]) -> List[str]:
        selected: Set[str] = set()

        for group_name in model_config.get("feature_groups", []):
            selected.update(self._resolve_group(group_name, available_features))

        selected.update(self._existing(model_config.get("include_features", []), available_features))
        selected.update(self._match_patterns(model_config.get("include_patterns", []), available_features))

        excluded = set(self._existing(model_config.get("exclude_features", []), available_features))
        excluded.update(self._match_patterns(model_config.get("exclude_patterns", []), available_features))

        if model_config.get("custom_features"):
            selected = set(self._existing(model_config["custom_features"], available_features))

        selected -= excluded
        selected = {feature for feature in selected if not feature.startswith("Price_H")}

        result = sorted(selected)
        if not result:
            raise ValueError(f"模型 {model_name} 没有选中任何可用特征，请检查 FeatureSelector 配置")

        logger.info("模型 %s 选中特征数: %s", model_name, len(result))
        return result

    def get_model_feature_info(self, model_name: str) -> Dict:
        model_config = self.get_model_features(model_name)
        return {
            "description": model_config.get("description", ""),
            "normalize": model_config.get("normalize", False),
            "feature_groups": model_config.get("feature_groups", []),
        }

    def list_all_models(self) -> List[str]:
        return list(self.config.get("model_features", {}).keys())

    def _resolve_group(self, group_name: str, available_features: List[str]) -> Set[str]:
        group = self.config.get("feature_groups", {}).get(group_name)
        if group is None:
            logger.warning("特征组不存在: %s", group_name)
            return set()
        selected = set(self._existing(group.get("features", []), available_features))
        selected.update(self._match_patterns(group.get("patterns", []), available_features))
        return selected

    @staticmethod
    def _existing(features: List[str], available_features: List[str]) -> List[str]:
        available = set(available_features)
        return [feature for feature in features if feature in available]

    @staticmethod
    def _match_patterns(patterns: List[str], available_features: List[str]) -> Set[str]:
        matched: Set[str] = set()
        for pattern in patterns:
            regex = re.compile(pattern)
            matched.update(feature for feature in available_features if regex.search(feature))
        return matched
