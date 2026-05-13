"""Runtime registry for forecasting strategies."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List


IMPLEMENTED_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "direct": {
        "name": "Direct",
        "status": "implemented",
        "description": "Train one independent single-output model for each forecast hour.",
        "feature_module": "EPF.feature_engineering.direct",
        "feature_engineer": "EPF.feature_engineering.direct:DirectFeatureEngineer",
        "train_module": "EPF.strategies.direct.train",
        "trainer": "EPF.strategies.direct.train:DirectTrainer",
        "evaluate_module": "EPF.strategies.direct.evaluate",
        "evaluator": "EPF.strategies.direct.evaluate:DirectEvaluator",
        "predict_module": "EPF.strategies.direct.predict",
        "predictor": "EPF.strategies.direct.predict:DirectPredictor",
        "backtest_module": "EPF.strategies.direct.backtest",
        "backtester": "EPF.strategies.direct.backtest:DirectMonthlyBacktester",
        "default_model": "lightgbm",
        "model_types": "EPF.models.factory:list_model_types",
    },
    "mimo": {
        "name": "MIMO",
        "status": "implemented",
        "description": "Train a true multi-output model that predicts the full 24-hour curve jointly.",
        "feature_module": "EPF.feature_engineering.mimo",
        "feature_engineer": "EPF.feature_engineering.mimo:MimoFeatureEngineer",
        "train_module": "EPF.strategies.mimo.train",
        "trainer": "EPF.strategies.mimo.train:MimoTrainer",
        "evaluate_module": "EPF.strategies.mimo.evaluate",
        "evaluator": "EPF.strategies.mimo.evaluate:MimoEvaluator",
        "predict_module": "EPF.strategies.mimo.predict",
        "predictor": "EPF.strategies.mimo.predict:MimoPredictor",
        "backtest_module": "EPF.strategies.mimo.backtest",
        "backtester": "EPF.strategies.mimo.backtest:MimoBacktester",
        "default_model": "tcn_mimo",
        "model_types": ["tcn_mimo"],
    },
}


PLANNED_STRATEGIES: Dict[str, Dict[str, str]] = {
    "recursive": {
        "name": "Recursive / Iterative",
        "status": "planned",
        "description": "Train one-step models and feed previous predictions into later horizons.",
    },
}


def list_strategies(include_planned: bool = True) -> Dict[str, Dict[str, Any]]:
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


def get_strategy_spec(strategy: str) -> Dict[str, Any]:
    ensure_implemented(strategy)
    return IMPLEMENTED_STRATEGIES[strategy]


def load_object(path: str) -> Any:
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def load_strategy_component(strategy: str, component: str) -> Any:
    spec = get_strategy_spec(strategy)
    if component not in spec:
        raise KeyError(f"策略 {strategy} 未注册组件: {component}")
    target = spec[component]
    if not isinstance(target, str) or ":" not in target:
        raise TypeError(f"策略 {strategy} 的组件 {component} 不是可加载对象路径")
    return load_object(target)


def strategy_model_types(strategy: str) -> List[str]:
    spec = get_strategy_spec(strategy)
    model_types = spec.get("model_types", [])
    if isinstance(model_types, str):
        model_types = load_object(model_types)()
    return sorted(model_types)


def all_model_types() -> List[str]:
    model_types = set()
    for strategy in implemented_strategy_names():
        model_types.update(strategy_model_types(strategy))
    return sorted(model_types)


def default_model_for_strategy(strategy: str) -> str:
    return str(get_strategy_spec(strategy)["default_model"])
