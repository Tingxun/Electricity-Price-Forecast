"""
特征工程脚本
从processed数据生成特征，保存到features目录

使用V2版本：按照实验方案要求，支持T+1预测场景
"""

import os
import sys
import logging
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from feature_engineering_v2 import FeatureEngineerV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    config = Config()
    
    logger.info("=" * 60)
    logger.info("开始生成特征 (V2 - T+1预测场景)")
    logger.info("=" * 60)
    
    # 检查processed数据是否存在
    processed_file = config.data_paths['processed'] / 'processed_data.csv'
    if not processed_file.exists():
        logger.error(f"processed数据不存在: {processed_file}")
        logger.error("请先运行 data_preprocessing.py 进行数据预处理")
        return
    
    # 加载processed数据
    logger.info(f"加载processed数据: {processed_file}")
    df = pd.read_csv(processed_file)
    df['日期'] = pd.to_datetime(df['日期'])
    logger.info(f"数据加载完成: {len(df)} 条记录")
    
    # 创建特征（使用V2版本）
    engineer = FeatureEngineerV2()
    features_df, target_cols = engineer.create_all_features(df)
    
    # 保存特征
    engineer.save_features(features_df, target_cols)
    
    logger.info("=" * 60)
    logger.info("特征生成完成")
    logger.info("=" * 60)
    logger.info(f"特征数据保存在: {config.data_paths['features']}")


if __name__ == '__main__':
    main()
