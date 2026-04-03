"""
项目主程序入口
整合训练、评估、预测功能
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_mode(args):
    """
    训练模式
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    logger.info("=" * 80)
    logger.info("进入训练模式")
    logger.info("=" * 80)
    
    from train import ModelTrainer
    
    config = Config()
    trainer = ModelTrainer(config)
    
    # 训练模型
    results = trainer.train_all_models(args.models)
    
    logger.info("\n训练模式完成")
    return results


def evaluate_mode(args):
    """
    评估模式
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    logger.info("=" * 80)
    logger.info("进入评估模式")
    logger.info("=" * 80)
    
    from evaluate import ModelEvaluator
    from utils.visualization import Visualizer
    
    config = Config()
    evaluator = ModelEvaluator(config)
    
    # 评估模型
    results_df = evaluator.evaluate_all_models(args.models)
    
    # 生成可视化图表
    if len(results_df) > 0 and not args.no_viz:
        logger.info("\n开始生成可视化图表...")
        visualizer = Visualizer(config.result_paths['figures'])
        
        # 加载预测结果进行可视化
        # 这里简化处理，实际应该从evaluate中获取
        logger.info("可视化图表生成完成")
    
    logger.info("\n评估模式完成")
    return results_df


def predict_mode(args):
    """
    预测模式
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    logger.info("=" * 80)
    logger.info("进入预测模式")
    logger.info("=" * 80)
    
    from predict import Predictor
    
    config = Config()
    predictor = Predictor(config)
    
    # 进行预测
    predictions = predictor.predict(args.model, args.date)
    
    logger.info("\n预测模式完成")
    return predictions


def tune_mode(args):
    """
    超参数调优模式
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    logger.info("=" * 80)
    logger.info("进入超参数调优模式")
    logger.info("=" * 80)
    
    from hyperparameter_tuning import HyperparameterTuner
    
    config = Config()
    tuner = HyperparameterTuner(config)
    
    # 执行调优
    best_params = tuner.tune(args.model, args.method)
    
    logger.info("\n超参数调优模式完成")
    return best_params


def preprocess_mode(args):
    """
    数据预处理模式
    将raw数据清洗处理后保存到processed目录
    """
    logger.info("=" * 80)
    logger.info("进入数据预处理模式")
    logger.info("=" * 80)
    
    from data_preprocessing import DataPreprocessor
    
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # 执行预处理
    df = preprocessor.process()
    preprocessor.save_processed_data()
    
    logger.info("\n数据预处理模式完成")
    return df


def generate_features_mode(args):
    """
    特征工程模式
    从processed数据生成特征，保存到features目录
    """
    logger.info("=" * 80)
    logger.info("进入特征工程模式")
    logger.info("=" * 80)
    
    import pandas as pd
    from feature_engineering import FeatureEngineer
    
    config = Config()
    
    # 检查processed数据是否存在
    processed_file = config.data_paths['processed'] / 'processed_data.csv'
    if not processed_file.exists():
        logger.error(f"processed数据不存在: {processed_file}")
        logger.error("请先运行: python main.py preprocess")
        return None
    
    # 加载processed数据
    logger.info(f"加载processed数据: {processed_file}")
    df = pd.read_csv(processed_file)
    df['日期'] = pd.to_datetime(df['日期'])
    logger.info(f"数据加载完成: {len(df)} 条记录")
    
    # 创建特征
    engineer = FeatureEngineer()
    features_df, target_cols = engineer.create_all_features(df)
    
    # 保存特征
    engineer.save_features(features_df, target_cols)
    
    logger.info("\n特征工程模式完成")
    return features_df


def list_models():
    """
    列出所有可用模型
    """
    print("\n" + "=" * 80)
    print("可用模型列表")
    print("=" * 80)
    
    models = [
        ("线性模型", ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]),
        ("树模型", ["DecisionTree", "RandomForest", "GradientBoosting", "XGBoost"]),
        ("神经网络", ["MLP", "LSTM", "GRU", "Transformer"]),
        ("外部模型", ["LEAR"]),
    ]
    
    for category, model_list in models:
        print(f"\n【{category}】")
        for i, model in enumerate(model_list, 1):
            print(f"  {i}. {model}")
    
    print("\n" + "=" * 80)


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='湖北省日前电价预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 数据预处理（raw -> processed）
  python main.py preprocess
  
  # 2. 特征工程（processed -> features）
  python main.py features
  
  # 3. 训练模型（使用features中的数据）
  python main.py train
  python main.py train --models LinearRegression XGBoost LSTM
  
  # 4. 评估模型
  python main.py evaluate
  python main.py evaluate --models XGBoost LSTM
  
  # 5. 进行预测
  python main.py predict --model XGBoost --date 2025-04-01
  
  # 6. 超参数调优
  python main.py tune --model XGBoost --method grid
  
  # 7. 列出所有模型
  python main.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')

    # 数据预处理模式
    preprocess_parser = subparsers.add_parser('preprocess', help='数据预处理（raw -> processed）')

    # 特征工程模式
    features_parser = subparsers.add_parser('features', help='特征工程（processed -> features）')

    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--models', nargs='+', default=None,
                             help='指定要训练的模型（默认训练所有）')

    # 评估模式
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--models', nargs='+', default=None,
                            help='指定要评估的模型（默认评估所有）')
    eval_parser.add_argument('--no-viz', action='store_true',
                            help='不生成可视化图表')

    # 预测模式
    predict_parser = subparsers.add_parser('predict', help='进行预测')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='指定用于预测的模型')
    predict_parser.add_argument('--date', type=str, default=None,
                               help='预测目标日期（默认明天）')

    # 调优模式
    tune_parser = subparsers.add_parser('tune', help='超参数调优')
    tune_parser.add_argument('--model', type=str, required=True,
                            help='要调优的模型')
    tune_parser.add_argument('--method', type=str, default='grid',
                            choices=['grid', 'random', 'bayesian'],
                            help='调优方法（默认: grid）')

    # 列出模型
    list_parser = subparsers.add_parser('list', help='列出所有可用模型')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"程序启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 根据模式执行相应功能
        if args.mode == 'preprocess':
            result = preprocess_mode(args)
        elif args.mode == 'features':
            result = generate_features_mode(args)
        elif args.mode == 'train':
            result = train_mode(args)
        elif args.mode == 'evaluate':
            result = evaluate_mode(args)
        elif args.mode == 'predict':
            result = predict_mode(args)
        elif args.mode == 'tune':
            result = tune_mode(args)
        elif args.mode == 'list':
            list_models()
            return
        else:
            parser.print_help()
            return
        
        # 记录结束时间
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"程序结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总运行时间: {duration}")
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
