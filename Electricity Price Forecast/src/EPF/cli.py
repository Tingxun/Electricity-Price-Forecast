"""Command line entry point for electricity price forecasting workflows."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from .config import Config
from .utils.strategy_registry import (
    all_model_types,
    all_strategy_names,
    default_model_for_strategy,
    ensure_implemented,
    list_strategies,
    load_strategy_component,
    strategy_model_types,
)


logger = logging.getLogger(__name__)

FORWARD_DEFAULT_START_MONTH = "2025-03"
FORWARD_DEFAULT_END_MONTH = "2025-06"


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "main.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _resolve_model(strategy: str, model_type: Optional[str]) -> str:
    if model_type is None:
        return default_model_for_strategy(strategy)
    supported = strategy_model_types(strategy)
    if model_type not in supported:
        raise ValueError(f"策略 {strategy} 不支持模型 {model_type}。可选模型: {', '.join(supported)}")
    return model_type


def features_mode(config: Config, strategy: str = "direct") -> None:
    ensure_implemented(strategy)

    import pandas as pd

    processed_file = config.get_data_path("processed_data")
    if not processed_file.exists():
        raise FileNotFoundError(f"预处理数据不存在: {processed_file}")

    df = pd.read_csv(processed_file)
    logger.info("加载预处理数据: %s, shape=%s", processed_file, df.shape)

    FeatureEngineer = load_strategy_component(strategy, "feature_engineer")
    engineer = FeatureEngineer()
    if strategy == "mimo":
        samples = engineer.create_features(df)
        engineer.save_features(samples, config.get_data_path("mimo_features"))
    else:
        hourly_results = engineer.create_all_features(df)
        engineer.save_features(hourly_results, config.get_data_path("direct_features"))


def train_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)
    args.model = _resolve_model(args.strategy, args.model)

    Trainer = load_strategy_component(args.strategy, "trainer")
    if args.strategy == "mimo":
        overrides = {
            "epochs": getattr(args, "epochs", None),
            "patience": getattr(args, "patience", None),
            "batch_size": getattr(args, "batch_size", None),
            "device": getattr(args, "device", None),
        }
        Trainer(
            config=config,
            model_type=args.model,
            test_months=args.test_months,
            model_config={k: v for k, v in overrides.items() if v is not None},
        ).train()
        return

    n_iter = 0 if getattr(args, "fixed_params", False) else args.n_iter
    Trainer(
        config=config,
        model_type=args.model,
        n_iter=n_iter,
        cv_folds=args.cv_folds,
        test_months=args.test_months,
    ).train(args.hours)


def evaluate_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)
    args.model = _resolve_model(args.strategy, args.model)

    Evaluator = load_strategy_component(args.strategy, "evaluator")
    results, overall = Evaluator(config, args.model, test_months=args.test_months).evaluate(args.hours)
    print(results.to_string(index=False))

    if args.strategy == "mimo":
        print(
            f"\nMAE={results['mae'].mean():.4f}, "
            f"RMSE={results['rmse'].mean():.4f}, "
            f"sMAPE={overall['overall_smape']:.2f}%, "
            f"AccRate={overall['overall_acc_rate']:.2f}%, "
        )
    else:
        print(
            f"\nMAE={results['mae'].mean():.4f}, "
            f"RMSE={results['rmse'].mean():.4f}, "
            f"sMAPE={results['smape'].mean():.2f}%, "
            f"AccRate={results['acc_rate'].mean():.2f}%, "
        )


def predict_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)
    args.model = _resolve_model(args.strategy, args.model)

    Predictor = load_strategy_component(args.strategy, "predictor")
    result = Predictor(config, args.model, test_months=args.test_months).predict(args.date)
    prices = result["predictions"]["prices"]
    print(f"Model: {args.model}")
    print(f"Target date: {result['target_date']}")
    if "feature_date_used" in result:
        print(f"Feature date used: {result['feature_date_used']}")
    print(f"Min/Max/Mean: {min(prices):.2f} / {max(prices):.2f} / {sum(prices) / len(prices):.2f}")


def backtest_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)
    args.model = _resolve_model(args.strategy, args.model)

    Backtester = load_strategy_component(args.strategy, "backtester")
    if args.strategy == "mimo":
        overrides = {
            "epochs": getattr(args, "epochs", None),
            "patience": getattr(args, "patience", None),
            "device": getattr(args, "device", None),
        }
        runner = Backtester(
            config=config,
            model_type=args.model,
            min_train_months=args.min_train_months,
            start_month=args.start_month,
            end_month=args.end_month,
            retrain_frequency=args.retrain_frequency,
            model_config={k: v for k, v in overrides.items() if v is not None},
        )
    else:
        runner = Backtester(
            config=config,
            model_type=args.model,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            min_train_months=args.min_train_months,
            start_month=args.start_month,
            end_month=args.end_month,
            retrain_frequency=args.retrain_frequency,
        )

    results = runner.run(args.hours)
    print(results.to_string(index=False))
    ok = results[results["status"] == "success"]
    if not ok.empty and "n_test" in ok:
        avg_smape = float((ok["smape"] * ok["n_test"]).sum() / ok["n_test"].sum())
    else:
        avg_smape = float(ok["smape"].mean())
    print(f"\nAverage sMAPE={avg_smape:.2f}%")


def list_mode() -> None:
    print("可运行策略:")
    for key, spec in list_strategies(include_planned=False).items():
        models = ", ".join(strategy_model_types(key))
        print(f"- {key}: {spec['description']} (models: {models})")

    planned = {key: spec for key, spec in list_strategies().items() if spec["status"] == "planned"}
    if planned:
        print("\n预留策略:")
        for key, spec in planned.items():
            print(f"- {key}: {spec['description']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="湖北日前电价预测统一工作流")
    subparsers = parser.add_subparsers(dest="mode")

    features_parser = subparsers.add_parser("features", help="生成策略特征")
    features_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    train_parser.add_argument("--model", default=None, choices=all_model_types(), help="模型类型")
    train_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定训练小时；MIMO 会忽略该参数")
    train_parser.add_argument("--n-iter", type=int, default=20, help="每小时随机搜索次数；0 表示默认参数")
    train_parser.add_argument("--fixed-params", action="store_true", help="跳过超参数调优，直接使用默认参数")
    train_parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    train_parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个")
    train_parser.add_argument("--epochs", type=int, default=None, help="MIMO 神经网络训练轮数")
    train_parser.add_argument("--patience", type=int, default=None, help="MIMO early stopping patience")
    train_parser.add_argument("--batch-size", type=int, default=None, help="MIMO batch size")
    train_parser.add_argument("--device", default=None, help="MIMO device: auto/cuda/cpu")

    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    eval_parser.add_argument("--model", default=None, choices=all_model_types(), help="模型类型")
    eval_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定评估小时")
    eval_parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个")

    predict_parser = subparsers.add_parser("predict", help="预测 24 小时价格")
    predict_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    predict_parser.add_argument("--model", default=None, choices=all_model_types(), help="模型类型")
    predict_parser.add_argument("--date", default=None, help="预测目标日期 YYYY-MM-DD")
    predict_parser.add_argument("--test-months", nargs="+", default=None, help="选择已训练模型对应的测试月份")

    backtest_parser = subparsers.add_parser("backtest", help="滚动回测模型")
    backtest_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    backtest_parser.add_argument("--model", default=None, choices=all_model_types(), help="模型类型")
    backtest_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定回测小时")
    backtest_parser.add_argument("--n-iter", type=int, default=0, help="每小时随机搜索次数；0 表示默认参数")
    backtest_parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    backtest_parser.add_argument("--min-train-months", type=int, default=3, help="开始回测前至少保留的训练月份数")
    backtest_parser.add_argument("--start-month", default=FORWARD_DEFAULT_START_MONTH, help="首个测试月份 YYYY-MM")
    backtest_parser.add_argument("--end-month", default=FORWARD_DEFAULT_END_MONTH, help="最后测试月份 YYYY-MM")
    backtest_parser.add_argument("--retrain-frequency", choices=["monthly", "weekly"], default="monthly", help="重训频率")
    backtest_parser.add_argument("--epochs", type=int, default=None, help="MIMO 神经网络训练轮数")
    backtest_parser.add_argument("--patience", type=int, default=None, help="MIMO early stopping patience")
    backtest_parser.add_argument("--device", default=None, help="MIMO device: auto/cuda/cpu")

    subparsers.add_parser("list", help="列出可运行策略和模型")

    run_all_parser = subparsers.add_parser("run-all", help="生成特征、训练并评估")
    run_all_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    run_all_parser.add_argument("--model", default=None, choices=all_model_types(), help="模型类型")
    run_all_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定小时")
    run_all_parser.add_argument("--n-iter", type=int, default=20, help="每小时随机搜索次数")
    run_all_parser.add_argument("--fixed-params", action="store_true", help="跳过超参数调优，直接使用默认参数")
    run_all_parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    run_all_parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个")
    run_all_parser.add_argument("--epochs", type=int, default=None, help="MIMO 神经网络训练轮数")
    run_all_parser.add_argument("--patience", type=int, default=None, help="MIMO early stopping patience")
    run_all_parser.add_argument("--batch-size", type=int, default=None, help="MIMO batch size")
    run_all_parser.add_argument("--device", default=None, help="MIMO device: auto/cuda/cpu")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config()

    try:
        if args.mode == "features":
            features_mode(config, args.strategy)
        elif args.mode == "train":
            train_mode(config, args)
        elif args.mode == "evaluate":
            evaluate_mode(config, args)
        elif args.mode == "predict":
            predict_mode(config, args)
        elif args.mode == "backtest":
            backtest_mode(config, args)
        elif args.mode == "list":
            list_mode()
        elif args.mode == "run-all":
            ensure_implemented(args.strategy)
            features_mode(config, args.strategy)
            train_mode(config, args)
            evaluate_mode(config, args)
        else:
            parser.print_help()
    except (NotImplementedError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
