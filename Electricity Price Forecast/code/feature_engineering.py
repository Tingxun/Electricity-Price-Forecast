"""
特征工程模块 
按照实验方案要求，支持T+1预测场景：
- 使用T-2及之前的历史实时数据构建滞后特征
- 使用T日披露的T+1日前数据作为输入特征
- 预测T+1的24点实时价格

数据格式：长格式（每天24行，每小时一行）
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler
import sys
import logging

sys.path.append(str(Path(__file__).parent))
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    特征工程类 
    按照实验方案实现T+1预测场景的特征构建
    
    预测场景：
    - 位于T时间点
    - 使用T-2及之前的历史实时数据构建滞后/滚动特征
    - 使用T日披露的T+1日前数据作为输入特征
    - 预测T+1的24点实时价格
    """
    
    def __init__(self):
        """初始化特征工程器"""
        # 特征配置
        self.feature_config = {
            'lag_periods': [1, 2, 3, 7],  # 滞后周期（天）
            'rolling_windows': [7, 14],  # 滚动窗口（天）
            'realtime_price_col': '平均出清价格-实时（元/MWh）',
            'dayahead_price_col': '平均出清价格-日前（元/MWh）',
        }
        
        # 实时数据列（用于构建滞后特征）
        self.realtime_cols = [
            '平均出清价格-实时（元/MWh）',
            '系统负荷-实时',
            '风电出力-实时',
            '光伏出力-实时',
            '水电出力-实时',
            '联络线计划-实时',
            '非市场化机组出力-实时',
        ]
        
        # 日前数据列（T日披露的T+1信息）
        self.dayahead_cols = [
            '系统负荷-日前',
            '风电出力-日前',
            '光伏出力-日前',
            '水电出力-日前',
            '联络线计划-日前',
            '非市场化机组出力-日前',
            '平均出清价格-日前（元/MWh）',
        ]
        
        # 列名映射（处理带单位和不带单位的情况）
        self.column_mapping = {
            '系统负荷-实时': '系统负荷-实时',
            '风电出力-实时': '风电出力-实时',
            '光伏出力-实时': '光伏出力-实时',
            '水电出力-实时': '水电出力-实时',
            '联络线计划-实时': '联络线计划-实时',
            '非市场化机组出力-实时': '非市场化机组出力-实时',
            '系统负荷-日前': '系统负荷-日前',
            '风电出力-日前': '风电出力-日前',
            '光伏出力-日前': '光伏出力-日前',
            '水电出力-日前': '水电出力-日前',
            '联络线计划-日前': '联络线计划-日前',
            '非市场化机组出力-日前': '非市场化机组出力-日前',
        }
        
        self.scaler = None
    
    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        创建所有特征（按实验方案要求）
        
        对于每一天T，构建特征用于预测T+1的24点价格：
        - 历史特征：使用T-2及之前的实时数据构建滞后和滚动特征（按小时）
        - 日前特征：使用T日披露的T+1日前数据（24小时）
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据（processed数据，长格式，每天24行）
            
        Returns
        -------
        features_df : pd.DataFrame
            特征数据框，每行代表一个预测样本（某天T，用于预测T+1的24点价格）
        target_cols : list
            目标列名列表（T+1的24点实时价格）
        """
        print("开始按实验方案创建特征...")
        print("数据格式：长格式（每天24小时，每小时一行）")
        
        # 确保数据按日期和小时排序
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(['日期', '小时']).reset_index(drop=True)
        
        # 获取所有日期
        all_dates = df['日期'].unique()
        print(f"数据时间范围: {all_dates.min()} 至 {all_dates.max()}")
        print(f"总天数: {len(all_dates)}")
        
        # 构建特征列表
        feature_list = []
        
        # 从第3天开始（需要至少2天历史数据）
        for i in range(2, len(all_dates)):
            target_date = all_dates[i]  # T+1
            current_date = all_dates[i-1]  # T
            
            # 获取历史数据（T-2及之前）
            historical_dates = all_dates[:i-1]  # T-2及之前
            historical_data = df[df['日期'].isin(historical_dates)].copy()
            
            # 获取当前日数据（T日，包含T+1的日前披露数据）
            current_data = df[df['日期'] == current_date].copy()
            if len(current_data) != 24:
                logger.warning(f"日期 {current_date} 数据不完整，跳过")
                continue
            current_data = current_data.sort_values('小时')
            
            # 获取目标日数据（T+1，用于提取目标价格）
            target_data = df[df['日期'] == target_date].copy()
            if len(target_data) != 24:
                logger.warning(f"目标日期 {target_date} 数据不完整，跳过")
                continue
            target_data = target_data.sort_values('小时')
            
            # 构建该样本的特征
            sample_features = self._build_sample_features(
                historical_data=historical_data,
                current_data=current_data,
                target_data=target_data,
                target_date=target_date
            )
            
            if sample_features is not None:
                feature_list.append(sample_features)
            
            if (i - 2) % 100 == 0 and (i - 2) > 0:
                print(f"进度: {i-2}/{len(all_dates)-2} - 处理日期 {target_date.date()}")
        
        if not feature_list:
            raise ValueError("未能生成任何有效特征")
        
        # 合并所有样本
        features_df = pd.DataFrame(feature_list)
        
        # 定义目标列（T+1的24点实时价格）
        target_cols = [f'Price_H{h:02d}' for h in range(24)]
        
        # 处理缺失值
        features_df = self._handle_missing_values(features_df)
        
        print(f"\n特征创建完成:")
        print(f"  总样本数: {len(features_df)}")
        print(f"  特征数: {len(features_df.columns) - len(target_cols)}")
        print(f"  目标数: {len(target_cols)}")
        print(f"  预测时间范围: {features_df['预测日期'].min().date()} 至 {features_df['预测日期'].max().date()}")
        
        return features_df, target_cols
    
    def _build_sample_features(self, historical_data: pd.DataFrame, 
                               current_data: pd.DataFrame,
                               target_data: pd.DataFrame,
                               target_date: pd.Timestamp) -> Optional[Dict]:
        """
        构建单个样本的特征
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            历史实时数据（T-2及之前，长格式）
        current_data : pd.DataFrame
            当前日数据（T日，包含T+1的日前披露数据，24小时）
        target_data : pd.DataFrame
            目标日数据（T+1，用于提取目标价格，24小时）
        target_date : pd.Timestamp
            目标日期（T+1）
            
        Returns
        -------
        features : dict or None
            特征字典
        """
        features = {}
        
        # 1. 时间特征（基于T+1日期）
        features['预测日期'] = target_date
        features['月份'] = target_date.month
        features['星期'] = target_date.dayofweek + 1
        features['是否周末'] = 1 if target_date.dayofweek >= 5 else 0
        features['季度'] = (target_date.month - 1) // 3 + 1
        
        # 2. 日前披露特征（T日披露的T+1信息，24小时）
        # 对于每个小时，添加日前预测值
        for _, row in current_data.iterrows():
            hour = int(row['小时'])
            for col in self.dayahead_cols:
                if col in row.index:
                    feature_name = col.replace('（MW）', '').replace('（元/MWh）', '')
                    features[f'日前_H{hour:02d}_{feature_name}'] = row[col]
        
        # 3. 历史滞后特征（基于T-2及之前的历史实时数据，按小时构建）
        if len(historical_data) > 0:
            # 为每个小时构建滞后特征
            for hour in range(24):
                hour_hist = historical_data[historical_data['小时'] == hour]
                
                if len(hour_hist) > 0:
                    # 按日期排序
                    hour_hist = hour_hist.sort_values('日期')
                    
                    for lag in self.feature_config['lag_periods']:
                        if len(hour_hist) >= lag:
                            lag_row = hour_hist.iloc[-lag]  # 倒数第lag天
                            for col in self.realtime_cols:
                                if col in lag_row.index:
                                    feature_name = col.replace('（MW）', '').replace('（元/MWh）', '')
                                    features[f'滞后{lag}天_H{hour:02d}_{feature_name}'] = lag_row[col]
        
        # 4. 历史滚动统计特征（基于T-2及之前的历史实时数据）
        if len(historical_data) >= 7 * 24:
            for hour in range(24):
                hour_hist = historical_data[historical_data['小时'] == hour]
                
                for col in self.realtime_cols:
                    if col in hour_hist.columns:
                        feature_name = col.replace('（MW）', '').replace('（元/MWh）', '')
                        
                        for window in self.feature_config['rolling_windows']:
                            if len(hour_hist) >= window:
                                values = hour_hist[col].tail(window)
                                features[f'滚动{window}天均值_H{hour:02d}_{feature_name}'] = values.mean()
                                features[f'滚动{window}天标准差_H{hour:02d}_{feature_name}'] = values.std()
        
        # 5. 目标值（T+1的24点实时价格）
        price_col = self.feature_config['realtime_price_col']
        target_data = target_data.sort_values('小时')
        
        for _, row in target_data.iterrows():
            hour = int(row['小时'])
            if price_col in row.index:
                features[f'Price_H{hour:02d}'] = row[price_col]
            else:
                features[f'Price_H{hour:02d}'] = np.nan
        
        # 检查是否有缺失的目标值
        for h in range(24):
            if f'Price_H{h:02d}' not in features or pd.isna(features[f'Price_H{h:02d}']):
                return None
        
        return features
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """处理缺失值"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = 0
                
                df[col] = df[col].fillna(fill_value)
                if missing_count > 0:
                    logger.debug(f"列 {col} 的 {missing_count} 个缺失值已填充")
        
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
            'feature_columns': [col for col in features_df.columns if col not in target_cols and col != '预测日期'],
            'n_samples': len(features_df),
            'n_features': len(features_df.columns) - len(target_cols) - 1,  # 减去预测日期
            'n_targets': len(target_cols),
            'description': 'T+1预测场景特征（长格式数据）'
        }
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        logger.info(f"特征信息已保存: {info_file}")
    
    def load_features(self, feature_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """加载特征数据"""
        if feature_path is None:
            feature_path = config.data_paths['features']
        
        data_file = os.path.join(feature_path, 'features.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"特征数据文件不存在: {data_file}")
        
        features_df = pd.read_csv(data_file)
        if '预测日期' in features_df.columns:
            features_df['预测日期'] = pd.to_datetime(features_df['预测日期'])
        
        logger.info(f"特征数据已加载: {len(features_df)} 条记录")
        
        import json
        info_file = os.path.join(feature_path, 'feature_info.json')
        with open(info_file, 'r', encoding='utf-8') as f:
            feature_info = json.load(f)
        target_cols = feature_info['target_columns']
        
        return features_df, target_cols


def main():
    """
    特征工程主函数
    从processed数据生成特征，保存到features目录
    """
    print("=" * 60)
    print("开始生成特征 (T+1预测场景)")
    print("=" * 60)
    
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
    
    # 创建特征
    engineer = FeatureEngineer()
    features_df, target_cols = engineer.create_all_features(df)
    
    # 保存特征
    engineer.save_features(features_df, target_cols)
    
    print("\n" + "=" * 60)
    print("特征生成完成")
    print("=" * 60)
    logger.info(f"特征数据保存在: {config.data_paths['features']}")


if __name__ == '__main__':
    main()
