"""
Command line entry point for the Direct electricity price forecasting workflow.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import Config
from .model_factory import list_model_types
from .utils.strategy_registry import all_strategy_names, ensure_implemented, list_strategies


logger = logging.getLogger(__name__)

FORWARD_DEFAULT_START_MONTH = "2025-03"
FORWARD_DEFAULT_END_MONTH = "2025-06"
MIMO_MODEL_TYPES = ["tcn_mimo"]


def all_model_types() -> List[str]:
    return sorted(set(list_model_types()) | set(MIMO_MODEL_TYPES))


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


def features_mode(config: Config, strategy: str = "direct") -> None:
    ensure_implemented(strategy)

    import pandas as pd

    processed_file = config.get_data_path("processed_data")
    if not processed_file.exists():
        raise FileNotFoundError(f"预处理数据不存在: {processed_file}")

    df = pd.read_csv(processed_file)
    logger.info("加载预处理数据: %s, shape=%s", processed_file, df.shape)

    if strategy == "mimo":
        from .feature_engineering_mimo import MimoFeatureEngineer

        engineer = MimoFeatureEngineer()
        samples = engineer.create_features(df)
        engineer.save_features(samples, config.get_data_path("mimo_features"))
        return

    from .feature_engineering_direct import DirectFeatureEngineer

    engineer = DirectFeatureEngineer()
    hourly_results = engineer.create_all_features(df)
    engineer.save_features(hourly_results, config.get_data_path("direct_features"))


def train_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)

    n_iter = 0 if getattr(args, "fixed_params", False) else args.n_iter
    if args.strategy == "mimo":
        from .train_mimo import MimoTrainer

        overrides = {
            "epochs": getattr(args, "epochs", None),
            "patience": getattr(args, "patience", None),
            "batch_size": getattr(args, "batch_size", None),
            "device": getattr(args, "device", None),
        }
        MimoTrainer(
            config=config,
            model_type=args.model,
            test_months=args.test_months,
            model_config={k: v for k, v in overrides.items() if v is not None},
        ).train()
        return

    from .train_direct import DirectTrainer

    trainer = DirectTrainer(
        config=config,
        model_type=args.model,
        n_iter=n_iter,
        cv_folds=args.cv_folds,
        test_months=args.test_months,
    )
    results = trainer.train(args.hours)


def evaluate_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)

    if args.strategy == "mimo":
        from .evaluate_mimo import MimoEvaluator

        results, overall = MimoEvaluator(config, args.model, test_months=args.test_months).evaluate(args.hours)
        print(results.to_string(index=False))
        print(
            f"\nMAE={results['mae'].mean():.4f}, "
            f"RMSE={results['rmse'].mean():.4f}, "
            f"sMAPE={overall['overall_smape']:.2f}%, "
            f"AccRate={overall['overall_acc_rate']:.2f}%, "
        )
        return

    from .evaluate_direct import DirectEvaluator

    results, overall_acc_rate = DirectEvaluator(config, args.model, test_months=args.test_months).evaluate(args.hours)
    print(results.to_string(index=False))
    print(
        f"\nMAE={results['mae'].mean():.4f}, "
        f"RMSE={results['rmse'].mean():.4f}, "
        f"sMAPE={results['smape'].mean():.2f}%, "
        f"AccRate={results['acc_rate'].mean():.2f}%, "
    )


def predict_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)

    if args.strategy == "mimo":
        from .predict_mimo import MimoPredictor

        result = MimoPredictor(config, args.model, test_months=args.test_months).predict(args.date)
        prices = result["predictions"]["prices"]
        print(f"Model: {args.model}")
        print(f"Target date: {result['target_date']}")
        print(f"Min/Max/Mean: {min(prices):.2f} / {max(prices):.2f} / {sum(prices) / len(prices):.2f}")
        return

    from .predict_direct import DirectPredictor

    result = DirectPredictor(config, args.model, test_months=args.test_months).predict(args.date)
    prices = result["predictions"]["prices"]
    print(f"Model: {args.model}")
    print(f"Target date: {result['target_date']}")
    print(f"Feature date used: {result['feature_date_used']}")
    print(f"Min/Max/Mean: {min(prices):.2f} / {max(prices):.2f} / {sum(prices) / len(prices):.2f}")


def backtest_mode(config: Config, args: argparse.Namespace) -> None:
    ensure_implemented(args.strategy)

    if args.strategy == "mimo":
        from .backtest_mimo import MimoBacktester

        overrides = {
            "epochs": getattr(args, "epochs", None),
            "patience": getattr(args, "patience", None),
            "device": getattr(args, "device", None),
        }
        results = MimoBacktester(
            config=config,
            model_type=args.model,
            min_train_months=args.min_train_months,
            start_month=args.start_month,
            end_month=args.end_month,
            retrain_frequency=args.retrain_frequency,
            model_config={k: v for k, v in overrides.items() if v is not None},
        ).run(args.hours)
        print(results.to_string(index=False))
        ok = results[results["status"] == "success"]
        avg_smape = float((ok["smape"] * ok["n_test"]).sum() / ok["n_test"].sum())
        print(f"\nAverage sMAPE={avg_smape:.2f}%")
        return

    from .backtest_direct import DirectMonthlyBacktester

    results = DirectMonthlyBacktester(
        config=config,
        model_type=args.model,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        min_train_months=args.min_train_months,
        start_month=args.start_month,
        end_month=args.end_month,
        retrain_frequency=args.retrain_frequency,
    ).run(args.hours)
    print(results.to_string(index=False))
    ok = results[results["status"] == "success"]
    if not ok.empty and "n_test" in ok:
        avg_smape = float((ok["smape"] * ok["n_test"]).sum() / ok["n_test"].sum())
    else:
        avg_smape = float(ok["smape"].mean())
    print(f"\nAverage sMAPE={avg_smape:.2f}%")

def list_mode() -> None:
    print("可用模型:")
    for model_type in list_model_types():
        print(f"- {model_type}")
    for model_type in MIMO_MODEL_TYPES:
        print(f"- {model_type}")
    print("\n预测策略:")
    for key, spec in list_strategies().items():
        print(f"- {key}: {spec['status']} - {spec['description']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="湖北日前电价预测 Direct 工作流")
    subparsers = parser.add_subparsers(dest="mode")

    features_parser = subparsers.add_parser("features", help="生成策略特征")
    features_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")

    train_parser = subparsers.add_parser("train", help="训练 Direct 每小时模型")
    train_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    train_parser.add_argument("--model", default="lightgbm", choices=all_model_types(), help="基模型类型")
    train_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定训练小时")
    train_parser.add_argument("--n-iter", type=int, default=20, help="每小时随机搜索次数；0 表示默认参数")
    train_parser.add_argument("--fixed-params", action="store_true", help="跳过超参数调优，直接使用模型工厂注册的默认固化参数")
    train_parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    train_parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个，默认使用最后一个可用月份")
    train_parser.add_argument("--epochs", type=int, default=None, help="MIMO 神经网络训练轮数")
    train_parser.add_argument("--patience", type=int, default=None, help="MIMO early stopping patience")
    train_parser.add_argument("--batch-size", type=int, default=None, help="MIMO batch size")
    train_parser.add_argument("--device", default=None, help="MIMO device: auto/cuda/cpu")

    eval_parser = subparsers.add_parser("evaluate", help="评估 Direct 模型")
    eval_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    eval_parser.add_argument("--model", default="lightgbm", choices=all_model_types(), help="基模型类型")
    eval_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定评估小时")
    eval_parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个，默认沿用模型训练月份或最后一个可用月份")

    predict_parser = subparsers.add_parser("predict", help="预测 24 小时价格")
    predict_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    predict_parser.add_argument("--model", default="lightgbm", choices=all_model_types(), help="基模型类型")
    predict_parser.add_argument("--date", default=None, help="预测目标日期 YYYY-MM-DD")
    predict_parser.add_argument("--test-months", nargs="+", default=None, help="选择用哪个测试月份训练出的模型；默认使用最新训练版本")

    backtest_parser = subparsers.add_parser("backtest", help="月份滚动回测 Direct 模型")
    backtest_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    backtest_parser.add_argument("--model", default="lightgbm", choices=all_model_types(), help="基模型类型")
    backtest_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定回测小时")
    backtest_parser.add_argument("--n-iter", type=int, default=0, help="每个小时随机搜索次数；0 表示默认参数")
    backtest_parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    backtest_parser.add_argument("--min-train-months", type=int, default=3, help="开始回测前至少保留的训练月份数")
    backtest_parser.add_argument("--start-month", default=FORWARD_DEFAULT_START_MONTH, help="首个测试月份 YYYY-MM")
    backtest_parser.add_argument("--end-month", default=FORWARD_DEFAULT_END_MONTH, help="最后测试月份 YYYY-MM")
    backtest_parser.add_argument(
        "--retrain-frequency",
        choices=["monthly", "weekly"],
        default="monthly",
        help="monthly 表示每个测试月训练一次；weekly 表示测试月内每 7 天重新训练一次",
    )
    backtest_parser.add_argument("--epochs", type=int, default=None, help="MIMO 神经网络训练轮数")
    backtest_parser.add_argument("--patience", type=int, default=None, help="MIMO early stopping patience")
    backtest_parser.add_argument("--device", default=None, help="MIMO device: auto/cuda/cpu")

    subparsers.add_parser("list", help="列出可用 Direct 基模型")

    run_all_parser = subparsers.add_parser("run-all", help="生成特征、训练并评估")
    run_all_parser.add_argument("--strategy", default="direct", choices=all_strategy_names(), help="预测策略")
    run_all_parser.add_argument("--model", default="lightgbm", choices=all_model_types(), help="基模型类型")
    run_all_parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定小时")
    run_all_parser.add_argument("--n-iter", type=int, default=20, help="每小时随机搜索次数")
    run_all_parser.add_argument("--fixed-params", action="store_true", help="跳过超参数调优，直接使用模型工厂注册的默认固化参数")
    run_all_parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    run_all_parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个，默认使用最后一个可用月份")

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
    except NotImplementedError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
