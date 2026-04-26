"""
项目主程序入口
整合特征工程、训练、评估、预测功能

数据预处理(data_preprocessing.ipynb)需用户手动运行
自动化流程从特征工程开始
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def features_mode(args):
    """
    特征工程模式
    从processed_data.csv生成特征，保存到features目录
    """
    logger.info("=" * 80)
    logger.info("进入特征工程模式")
    logger.info("=" * 80)
    
    import pandas as pd
    from feature_engineering import FeatureEngineer
    
    config = Config()
    
    # 检查预处理数据是否存在
    processed_file = config.get_data_path('processed_data')
    if not processed_file.exists():
        logger.error(f"预处理数据不存在: {processed_file}")
        logger.error("请先运行 data_preprocessing.ipynb 完成数据预处理")
        return None
    
    # 加载预处理数据
    logger.info(f"加载预处理数据: {processed_file}")
    df = pd.read_csv(processed_file)
    logger.info(f"数据加载完成: {df.shape}")
    
    # 创建特征
    engineer = FeatureEngineer()
    features_df, target_cols = engineer.create_all_features(df)
    
    # 保存特征
    features_file = config.get_data_path('features') / 'features.csv'
    features_df.to_csv(features_file, index=False)
    logger.info(f"特征已保存: {features_file}")
    
    # 保存特征信息
    import json
    feature_info = {
        'target_columns': target_cols,
        'feature_columns': [c for c in features_df.columns if c not in target_cols and c != '预测日期'],
        'n_samples': len(features_df),
        'n_features': len(features_df.columns) - len(target_cols) - 1,
        'n_targets': len(target_cols),
        'created_at': datetime.now().isoformat()
    }
    with open(config.get_data_path('feature_info'), 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    
    logger.info("\n特征工程模式完成")
    return features_df


def train_mode(args):
    """
    训练模式
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
    """
    logger.info("=" * 80)
    logger.info("进入评估模式")
    logger.info("=" * 80)
    
    from evaluate import ModelEvaluator
    
    config = Config()
    evaluator = ModelEvaluator(config)
    
    # 评估模型
    results_df = evaluator.evaluate_all_models(args.models)
    
    logger.info("\n评估模式完成")
    return results_df


def predict_mode(args):
    """
    预测模式
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


def run_all_mode(args):
    """
    全自动模式：特征工程 -> 训练 -> 评估
    """
    logger.info("=" * 80)
    logger.info("进入全自动模式")
    logger.info("=" * 80)
    
    # 1. 特征工程
    features_df = features_mode(args)
    if features_df is None:
        return None
    
    # 2. 训练
    train_mode(args)
    
    # 3. 评估
    evaluate_mode(args)
    
    logger.info("\n全自动模式完成")


def list_models():
    """
    列出所有可用模型
    """
    print("\n" + "=" * 80)
    print("可用模型列表")
    print("=" * 80)
    
    models = [
        ("线性模型", ["LinearRegression", "Ridge", "Lasso"]),
        ("树模型", ["RandomForest", "XGBoost", "LightGBM"]),
        ("神经网络", ["MLP"]),
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
        description='湖北省日前电价预测系统 - 自动化流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 特征工程（从processed_data.csv生成特征）
  python main.py features
  
  # 2. 训练模型
  python main.py train
  python main.py train --models Ridge XGBoost
  
  # 3. 评估模型
  python main.py evaluate
  
  # 4. 全自动模式（特征工程 -> 训练 -> 评估）
  python main.py run-all
  
  # 5. 进行预测
  python main.py predict --model XGBoost --date 2025-04-01
  
  # 6. 列出所有模型
  python main.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')

    # 特征工程模式
    features_parser = subparsers.add_parser('features', help='特征工程')

    # 训练模式
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--models', nargs='+', default=None,
                             help='指定要训练的模型（默认训练所有）')

    # 评估模式
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--models', nargs='+', default=None,
                            help='指定要评估的模型（默认评估所有）')

    # 预测模式
    predict_parser = subparsers.add_parser('predict', help='进行预测')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='指定用于预测的模型')
    predict_parser.add_argument('--date', type=str, default=None,
                               help='预测目标日期（默认明天）')

    # 全自动模式
    run_all_parser = subparsers.add_parser('run-all', help='全自动模式（特征工程->训练->评估）')
    run_all_parser.add_argument('--models', nargs='+', default=None,
                               help='指定要训练的模型')

    # 列出模型
    list_parser = subparsers.add_parser('list', help='列出所有可用模型')

    args = parser.parse_args()

    if args.mode == 'features':
        features_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'run-all':
        run_all_mode(args)
    elif args.mode == 'list':
        list_models()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
