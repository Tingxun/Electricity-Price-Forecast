"""
Feature selection for Direct multi-step models.

The selected feature set is controlled by ``feature_config.yaml``. Feature
groups may contain exact feature names and regular-expression patterns, which
keeps the configuration compact for hourly/generated column names.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select model-specific features from available columns."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "feature_config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_model_features(self, model_name: str) -> Dict:
        model_features = self.config.get("model_features", {})
        return model_features.get(model_name, self.config.get("default", {}))

    def select_features_for_model(self, model_name: str, available_features: List[str]) -> List[str]:
        model_config = self.get_model_features(model_name)
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
        selected = {f for f in selected if not f.startswith("Price_H")}

        result = sorted(selected)
        if not result:
            raise ValueError(f"模型 {model_name} 没有选中任何可用特征，请检查 feature_config.yaml")

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

        if isinstance(group, list):
            return set(self._existing(group, available_features))

        features = set(self._existing(group.get("features", []), available_features))
        features.update(self._match_patterns(group.get("patterns", []), available_features))
        return features

    @staticmethod
    def _existing(features: Iterable[str], available_features: List[str]) -> List[str]:
        available = set(available_features)
        return [feature for feature in features if feature in available]

    @staticmethod
    def _match_patterns(patterns: Iterable[str], available_features: List[str]) -> List[str]:
        matched: Set[str] = set()
        for pattern in patterns:
            regex = re.compile(pattern)
            matched.update(feature for feature in available_features if regex.search(feature))
        return sorted(matched)


def main() -> None:
    selector = FeatureSelector()
    print("Configured Direct models:")
    for model_name in selector.list_all_models():
        info = selector.get_model_feature_info(model_name)
        print(f"- {model_name}: {info['description']}")


if __name__ == "__main__":
    main()
