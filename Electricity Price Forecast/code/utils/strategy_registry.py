"""
Forecasting strategy registry.

Direct is the only implemented strategy today. Recursive and MIMO are reserved
as explicit future strategy slots so the command line, documentation, and output
layout can evolve without another structural rewrite.
"""

from typing import Dict, List


IMPLEMENTED_STRATEGIES: Dict[str, Dict[str, str]] = {
    "direct": {
        "name": "Direct",
        "status": "implemented",
        "description": "Train one independent single-output model for each forecast hour.",
        "feature_module": "feature_engineering_direct.py",
        "train_module": "train_direct.py",
        "evaluate_module": "evaluate_direct.py",
        "predict_module": "predict_direct.py",
    }
}


PLANNED_STRATEGIES: Dict[str, Dict[str, str]] = {
    "recursive": {
        "name": "Recursive / Iterative",
        "status": "planned",
        "description": "Train one-step models and feed previous predictions into later horizons.",
    },
    "mimo": {
        "name": "MIMO",
        "status": "planned",
        "description": "Train a true multi-output model that predicts the full 24-hour curve jointly.",
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
