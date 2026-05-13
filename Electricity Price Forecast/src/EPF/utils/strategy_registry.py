"""
Forecasting strategy registry.

Direct and MIMO are implemented. Recursive is reserved as a future strategy slot.
"""

from typing import Dict, List


IMPLEMENTED_STRATEGIES: Dict[str, Dict[str, str]] = {
    "direct": {
        "name": "Direct",
        "status": "implemented",
        "description": "Train one independent single-output model for each forecast hour.",
        "feature_module": "EPF.feature_engineering.direct",
        "train_module": "EPF.strategies.direct.train",
        "evaluate_module": "EPF.strategies.direct.evaluate",
        "predict_module": "EPF.strategies.direct.predict",
    },
    "mimo": {
        "name": "MIMO",
        "status": "implemented",
        "description": "Train a true multi-output model that predicts the full 24-hour curve jointly.",
        "feature_module": "EPF.feature_engineering.mimo",
        "train_module": "EPF.strategies.mimo.train",
        "evaluate_module": "EPF.strategies.mimo.evaluate",
        "predict_module": "EPF.strategies.mimo.predict",
    }
}


PLANNED_STRATEGIES: Dict[str, Dict[str, str]] = {
    "recursive": {
        "name": "Recursive / Iterative",
        "status": "planned",
        "description": "Train one-step models and feed previous predictions into later horizons.",
    },
}


def list_strategies(include_planned: bool = True) -> Dict[str, Dict[str, str]]:
    strategies = IMPLEMENTED_STRATEGIES.copy()
    if include_planned:
        strategies.update(PLANNED_STRATEGIES)
    return strategies


def implemented_strategy_names() -> List[str]:
    return sorted(IMPLEMENTED_STRATEGIES.keys())


def all_strategy_names() -> List[str]:
    return sorted(list_strategies(include_planned=True).keys())


def ensure_implemented(strategy: str) -> None:
    if strategy in IMPLEMENTED_STRATEGIES:
        return

    if strategy in PLANNED_STRATEGIES:
        raise NotImplementedError(
            f"策略 {strategy} 已预留但尚未实现。当前可运行策略: {', '.join(implemented_strategy_names())}"
        )

    raise ValueError(f"未知策略: {strategy}. 可选策略: {', '.join(all_strategy_names())}")
