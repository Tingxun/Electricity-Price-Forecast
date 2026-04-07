"""
特征工程模块 

可用特征：
- 时间特征：月份、星期、是否周末、季度、小时、是否高峰时段、是否夜间
- 日前特征：系统负荷-日前、非市场化机组出力-日前、新能源出力-日前、平均出清价格-日前
- 目标：平均出清价格-实时
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys
import logging

sys.path.append(str(Path(__file__).parent))
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    特征工程类 
    
    预测场景：T时刻预测T+1的24点实时价格
    - 使用T-2及之前的历史实时价格构建滞后特征
    - 使用T日披露的T+1日前数据作为输入特征
    """
    
    def __init__(self):
        # 滞后周期配置
        self.lag_periods = [1, 2, 3, 7]  # 1天、2天、3天、7天前
        
        # 日前披露特征列
        self.dayahead_cols = [
            '系统负荷-日前',
            '非市场化机组出力-日前',
            '新能源出力-日前',
            '平均出清价格-日前（元/MWh）',
        ]
        
        # 实时价格列（用于滞后特征和目标）
        self.price_col = '平均出清价格-实时（元/MWh）'
    
    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        创建所有特征
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据（processed数据，长格式，每天24行）
            
        Returns
        -------
        features_df : pd.DataFrame
            特征数据框，每行代表一个预测样本
        target_cols : list
            目标列名列表（T+1的24点实时价格）
        """
        print("开始创建特征...")
        
        # 数据准备
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(['日期', '小时']).reset_index(drop=True)
        
        # 获取所有日期
        all_dates = df['日期'].unique()
        print(f"数据时间范围: {all_dates.min()} 至 {all_dates.max()}")
        print(f"总天数: {len(all_dates)}")
        
        feature_list = []
        
        # 从第3天开始（需要至少2天历史数据）
        for i in range(2, len(all_dates)):
            target_date = all_dates[i]  # T+1
            current_date = all_dates[i-1]  # T
            
            # 获取历史数据（T-2及之前）
            historical_dates = all_dates[:i-1]
            historical_data = df[df['日期'].isin(historical_dates)].copy()
            
            # 获取当前日数据（T日，包含T+1的日前披露数据）
            current_data = df[df['日期'] == current_date].copy()
            if len(current_data) != 24:
                continue
            current_data = current_data.sort_values('小时')
            
            # 获取目标日数据（T+1）
            target_data = df[df['日期'] == target_date].copy()
            if len(target_data) != 24:
                continue
            target_data = target_data.sort_values('小时')
            
            # 构建特征
            sample_features = self._build_sample_features(
                historical_data=historical_data,
                current_data=current_data,
                target_data=target_data,
                target_date=target_date
            )
            
            if sample_features is not None:
                feature_list.append(sample_features)
            
            if (i - 2) % 50 == 0 and (i - 2) > 0:
                print(f"进度: {i-2}/{len(all_dates)-2}")
        
        if not feature_list:
            raise ValueError("未能生成任何有效特征")
        
        features_df = pd.DataFrame(feature_list)
        
        # 目标列：T+1的24点实时价格
        target_cols = [f'Price_H{h:02d}' for h in range(24)]
        
        # 处理缺失值
        features_df = self._handle_missing_values(features_df)
        
        print(f"\n特征创建完成:")
        print(f"  总样本数: {len(features_df)}")
        print(f"  特征数: {len(features_df.columns) - len(target_cols) - 1}")  # 减去预测日期
        print(f"  目标数: {len(target_cols)}")
        
        return features_df, target_cols
    
    def _build_sample_features(self, historical_data: pd.DataFrame,
                               current_data: pd.DataFrame,
                               target_data: pd.DataFrame,
                               target_date: pd.Timestamp) -> Optional[Dict]:
        """构建单个样本的特征"""
        features = {}
        
        # 1. 时间特征
        features['预测日期'] = target_date
        features['月份'] = target_date.month
        features['星期'] = target_date.dayofweek + 1
        features['是否周末'] = 1 if target_date.dayofweek >= 5 else 0
        features['季度'] = (target_date.month - 1) // 3 + 1
        
        # 2. 日前披露特征（24小时）
        for _, row in current_data.iterrows():
            hour = int(row['小时'])
            for col in self.dayahead_cols:
                if col in row.index:
                    # 简化列名
                    short_name = col.replace('（元/MWh）', '').replace('-日前', '')
                    features[f'日前_H{hour:02d}_{short_name}'] = row[col]
        
        # 3. 历史滞后特征（各滞后周期的24小时价格）
        unique_hist_dates = historical_data['日期'].unique()
        for lag in self.lag_periods:
            if len(unique_hist_dates) >= lag:
                lag_date = unique_hist_dates[-lag]
                lag_data = historical_data[historical_data['日期'] == lag_date]
                if len(lag_data) == 24:
                    lag_data = lag_data.sort_values('小时')
                    for _, row in lag_data.iterrows():
                        hour = int(row['小时'])
                        if self.price_col in row.index:
                            features[f'滞后{lag}天_H{hour:02d}_价格'] = row[self.price_col]
        
        # 4. 目标值（T+1的24点实时价格）
        target_data = target_data.sort_values('小时')
        for _, row in target_data.iterrows():
            hour = int(row['小时'])
            if self.price_col in row.index and pd.notna(row[self.price_col]):
                features[f'Price_H{hour:02d}'] = row[self.price_col]
            else:
                return None  # 目标值缺失，跳过此样本
        
        return features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值 - 使用中位数填充"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def save_features(self, features_df: pd.DataFrame, target_cols: List[str],
                     feature_path: Optional[str] = None):
        """保存特征数据"""
        if feature_path is None:
            feature_path = config.data_paths['features']
        
        os.makedirs(feature_path, exist_ok=True)
        
        # 保存特征数据
        data_file = os.path.join(feature_path, 'features.csv')
        features_df.to_csv(data_file, index=False, encoding='utf-8-sig')
        logger.info(f"特征数据已保存: {data_file}")
        
        # 保存特征信息
        import json
        info_file = os.path.join(feature_path, 'feature_info.json')
        feature_info = {
            'target_columns': target_cols,
            'feature_columns': [c for c in features_df.columns if c not in target_cols and c != '预测日期'],
            'n_samples': len(features_df),
            'n_features': len(features_df.columns) - len(target_cols) - 1,
            'n_targets': len(target_cols),
        }
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        logger.info(f"特征信息已保存: {info_file}")
    
    def load_features(self, feature_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """加载特征数据"""
        if feature_path is None:
            feature_path = config.data_paths['features']
        
        data_file = os.path.join(feature_path, 'features.csv')
        features_df = pd.read_csv(data_file)
        if '预测日期' in features_df.columns:
            features_df['预测日期'] = pd.to_datetime(features_df['预测日期'])
        
        import json
        info_file = os.path.join(feature_path, 'feature_info.json')
        with open(info_file, 'r', encoding='utf-8') as f:
            feature_info = json.load(f)
        
        return features_df, feature_info['target_columns']


def main():
    """特征工程主函数"""
    print("=" * 60)
    print("开始生成特征")
    print("=" * 60)
    
    # 检查processed数据
    processed_file = config.data_paths['processed'] / 'processed_data.csv'
    if not processed_file.exists():
        logger.error(f"processed数据不存在: {processed_file}")
        return
    
    # 加载数据
    df = pd.read_csv(processed_file)
    df['日期'] = pd.to_datetime(df['日期'])
    logger.info(f"数据加载完成: {len(df)} 条记录")
    
    # 创建特征
    engineer = FeatureEngineer()
    features_df, target_cols = engineer.create_all_features(df)
    
    # 保存特征
    engineer.save_features(features_df, target_cols)
    
    print("\n" + "=" * 60)
    print("特征生成完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
