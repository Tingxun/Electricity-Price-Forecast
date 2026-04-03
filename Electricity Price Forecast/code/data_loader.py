"""
数据加载模块
负责加载原始数据文件

注意：数据预处理、特征工程、质量检查等功能已迁移至：
- data_preprocessing.py: 数据清洗和预处理
- feature_engineering.py: 特征工程
- train.py: 数据划分和训练准备
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent))
from config import config


class DataLoader:
    """
    数据加载器
    简化版：仅负责从指定路径加载原始数据
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化数据加载器
        
        Parameters
        ----------
        data_path : str, optional
            数据文件路径，默认使用config中的raw路径
        """
        if data_path is None:
            self.data_path = config.data_paths['raw'] / '市场边界_出清价格总表.csv'
        else:
            self.data_path = Path(data_path)
        
        self.raw_data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns
        -------
        df : pd.DataFrame
            加载的数据，包含解析后的日期时间列
        """
        print(f"正在加载数据: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # 转换日期列
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 创建完整的时间戳
        df['datetime'] = pd.to_datetime(
            df['日期'].astype(str) + ' ' + df['时段'].str.split('-').str[0]
        )
        
        # 按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        self.raw_data = df.copy()
        
        print(f"数据加载完成，共 {len(df)} 条记录")
        print(f"时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
        
        return df
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        加载预处理后的数据
        
        Returns
        -------
        df : pd.DataFrame
            预处理后的数据
        """
        processed_path = config.data_paths['processed'] / 'processed_data.csv'
        
        if not processed_path.exists():
            raise FileNotFoundError(
                f"预处理数据不存在: {processed_path}\n"
                f"请先运行: python main.py preprocess"
            )
        
        print(f"正在加载预处理数据: {processed_path}")
        df = pd.read_csv(processed_path)
        df['日期'] = pd.to_datetime(df['日期'])
        
        print(f"预处理数据加载完成，共 {len(df)} 条记录")
        return df


def main():
    """测试数据加载器"""
    print("=== 数据加载器测试 ===")
    
    loader = DataLoader()
    
    # 测试加载原始数据
    print("\n1. 测试加载原始数据:")
    df_raw = loader.load_data()
    print(f"列名: {list(df_raw.columns[:5])}...")
    
    # 测试加载预处理数据
    print("\n2. 测试加载预处理数据:")
    try:
        df_processed = loader.load_processed_data()
        print(f"列名: {list(df_processed.columns[:5])}...")
    except FileNotFoundError as e:
        print(f"提示: {e}")
    
    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    main()
