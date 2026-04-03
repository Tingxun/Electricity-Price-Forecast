"""
数据预处理脚本
将raw数据清洗处理后保存到processed目录
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    数据预处理器
    负责数据清洗、缺失值处理、异常值检测等
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.raw_data = None
        self.processed_data = None
        
        # 确保processed目录存在
        os.makedirs(config.data_paths['processed'], exist_ok=True)
        
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        raw_path = self.config.data_paths['raw'] / '市场边界_出清价格总表.csv'
        logger.info(f"加载原始数据: {raw_path}")
        
        df = pd.read_csv(raw_path)
        logger.info(f"原始数据加载完成: {len(df)} 条记录")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        - 转换日期格式
        - 处理重复值
        - 基础数据类型转换
        """
        logger.info("开始数据清洗...")
        df = df.copy()
        
        # 转换日期列
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 创建完整的时间戳
        df['datetime'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['时段'].str.split('-').str[0])
        
        # 按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 删除重复值（基于日期和时段）
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['日期', '时段'], keep='first')
        after_dedup = len(df)
        if before_dedup != after_dedup:
            logger.info(f"删除重复值: {before_dedup - after_dedup} 条")
        
        logger.info(f"数据清洗完成: {len(df)} 条记录")
        logger.info(f"时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
        """
        处理缺失值
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        strategy : str
            缺失值处理策略: 'interpolate', 'median', 'mean', 'forward'
        """
        logger.info(f"处理缺失值 (策略: {strategy})...")
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if strategy == 'interpolate':
                    # 时间序列插值
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'forward':
                    df[col] = df[col].ffill().bfill()
                
                logger.info(f"  {col}: {missing_count} 个缺失值已填充")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        处理异常值
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        method : str
            异常值处理方法: 'iqr', 'zscore', 'clip'
        """
        logger.info(f"处理异常值 (方法: {method})...")
        df = df.copy()
        
        # 需要处理异常值的列
        outlier_cols = [
            '系统负荷-实时', '系统负荷-日前',
            '风电出力-实时', '风电出力-日前',
            '光伏出力-实时', '光伏出力-日前',
            '平均出清价格-实时（元/MWh）', '平均出清价格-日前（元/MWh）'
        ]
        
        for col in outlier_cols:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 使用3倍IQR，保留更多数据
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                if outliers > 0:
                    logger.info(f"  {col}: {outliers} 个异常值已处理")
                    
            elif method == 'clip':
                # 使用分位数截断
                lower = df[col].quantile(0.001)
                upper = df[col].quantile(0.999)
                df[col] = df[col].clip(lower, upper)
        
        return df
    
    def process(self) -> pd.DataFrame:
        """
        执行完整的数据预处理流程
        
        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        logger.info("=" * 60)
        logger.info("开始数据预处理")
        logger.info("=" * 60)
        
        # 1. 加载原始数据
        df = self.load_raw_data()
        
        # 2. 数据清洗
        df = self.clean_data(df)
        
        # 3. 处理缺失值
        df = self.handle_missing_values(df, strategy='interpolate')
        
        # 4. 处理异常值
        df = self.handle_outliers(df, method='iqr')
        
        self.processed_data = df
        
        logger.info("=" * 60)
        logger.info("数据预处理完成")
        logger.info("=" * 60)
        
        return df
    
    def save_processed_data(self, filename: str = 'processed_data.csv'):
        """
        保存处理后的数据
        
        Parameters
        ----------
        filename : str
            保存文件名
        """
        if self.processed_data is None:
            raise ValueError("没有可保存的数据，请先执行process()")
        
        save_path = self.config.data_paths['processed'] / filename
        self.processed_data.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"处理后的数据已保存: {save_path}")
        
        # 保存数据信息
        info = {
            'total_records': len(self.processed_data),
            'date_range': {
                'start': str(self.processed_data['日期'].min()),
                'end': str(self.processed_data['日期'].max())
            },
            'columns': list(self.processed_data.columns),
            'numeric_columns': list(self.processed_data.select_dtypes(include=[np.number]).columns)
        }
        
        import json
        info_path = self.config.data_paths['processed'] / 'data_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"数据信息已保存: {info_path}")


def main():
    """主函数"""
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # 执行预处理
    df = preprocessor.process()
    
    # 保存结果
    preprocessor.save_processed_data()
    
    logger.info("\n预处理完成！")
    logger.info(f"处理后的数据保存在: {config.data_paths['processed']}")


if __name__ == '__main__':
    main()
