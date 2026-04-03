"""
特征工程模块
负责构建电价预测所需的各类特征

根据实验方案要求，支持：
- 时间特征构建
- 历史滞后特征构建
- 滚动统计特征构建
- 日前披露特征处理
- 衍生特征构建
- 特征标准化和预处理
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import sys
import logging

sys.path.append(str(Path(__file__).parent))
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    特征工程类
    负责构建电价预测所需的各类特征
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化特征工程器
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径
        """
        if config_path:
            # 如果有自定义配置，可以在这里加载
            pass
        
        # 特征配置
        self.feature_config = {
            'lag_periods': [1, 2, 3, 7],  # 滞后周期
            'rolling_windows': [7, 14, 30],  # 滚动窗口
            'target_column': '平均出清价格-实时（元/MWh）',
            'time_columns': ['年', '月', '日', '星期', '是否周末', '季度', '小时', '是否高峰时段', '是否夜间']
        }
        
        # 预处理器
        self.scaler = None
        self.feature_selector = None
        
        # 特征重要性
        self.feature_importance = {}
    
    def create_time_features(self, df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        """
        构建时间特征
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        target_date : pd.Timestamp
            目标日期（T+1）
            
        Returns
        -------
        time_features : pd.DataFrame
            时间特征
        """
        time_features = df.copy()
        
        # 确保有日期列
        if '日期' not in time_features.columns:
            time_features['日期'] = target_date
        
        # 提取详细时间特征
        time_features['小时'] = time_features['小时'].astype(int)
        time_features['星期'] = time_features['星期'].astype(int)
        time_features['月份'] = time_features['月'].astype(int)
        
        # 季节特征（1-4季）
        time_features['季节'] = ((time_features['月份'] - 1) // 3 + 1).astype(int)
        
        # 是否工作日（星期1-5为工作日）
        time_features['是否工作日'] = ((time_features['星期'] >= 1) & (time_features['星期'] <= 5)).astype(int)
        
        # 节假日特征（简化版，实际应用中需要节假日日历）
        time_features['是否节假日'] = 0  # 占位符，需要外部节假日数据
        
        # 时间段特征
        time_features['是否上午'] = ((time_features['小时'] >= 6) & (time_features['小时'] <= 12)).astype(int)
        time_features['是否下午'] = ((time_features['小时'] >= 13) & (time_features['小时'] <= 18)).astype(int)
        time_features['是否晚上'] = ((time_features['小时'] >= 19) & (time_features['小时'] <= 23)).astype(int)
        
        # 正弦余弦编码小时特征（处理周期性）
        time_features['小时_sin'] = np.sin(2 * np.pi * time_features['小时'] / 24)
        time_features['小时_cos'] = np.cos(2 * np.pi * time_features['小时'] / 24)
        
        # 正弦余弦编码星期特征
        time_features['星期_sin'] = np.sin(2 * np.pi * time_features['星期'] / 7)
        time_features['星期_cos'] = np.cos(2 * np.pi * time_features['星期'] / 7)
        
        print(f"时间特征构建完成，共 {len(time_features.columns)} 个特征")
        
        return time_features
    
    def create_lag_features(self, historical_data: pd.DataFrame, 
                           target_column: str = None) -> pd.DataFrame:
        """
        构建滞后特征
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            历史实时数据（T-2及之前）
        target_column : str, optional
            目标列名，默认使用配置中的目标列
            
        Returns
        -------
        lag_features : pd.DataFrame
            滞后特征
        """
        if target_column is None:
            target_column = self.feature_config['target_column']
        
        lag_features = historical_data.copy()
        
        # 按日期和小时排序
        lag_features = lag_features.sort_values(['日期', '小时'])
        
        # 创建滞后特征
        for lag in self.feature_config['lag_periods']:
            # 价格滞后特征（同一小时，前lag天）
            lag_features[f'价格滞后_{lag}天'] = lag_features.groupby('小时')[target_column].shift(lag * 24)
            
            # 负荷滞后特征
            if '系统负荷-实时' in lag_features.columns:
                lag_features[f'负荷滞后_{lag}天'] = lag_features.groupby('小时')['系统负荷-实时'].shift(lag * 24)
        
        # 价格变化率特征
        lag_features['价格变化率_1小时'] = lag_features.groupby('日期')[target_column].pct_change()
        lag_features['价格变化率_24小时'] = lag_features.groupby('日期')[target_column].pct_change(periods=24)
        
        print(f"滞后特征构建完成，共 {len([col for col in lag_features.columns if '滞后' in col or '变化率' in col])} 个滞后特征")
        
        return lag_features
    
    def create_rolling_features(self, historical_data: pd.DataFrame,
                              target_column: str = None) -> pd.DataFrame:
        """
        构建滚动统计特征
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            历史实时数据
        target_column : str, optional
            目标列名
            
        Returns
        -------
        rolling_features : pd.DataFrame
            滚动统计特征
        """
        if target_column is None:
            target_column = self.feature_config['target_column']
        
        rolling_features = historical_data.copy()
        
        # 按日期和小时排序
        rolling_features = rolling_features.sort_values(['日期', '小时'])
        
        # 为每个小时构建滚动特征
        for window in self.feature_config['rolling_windows']:
            # 同一小时的滚动统计
            rolling_features[f'价格滚动均值_{window}天'] = rolling_features.groupby('小时')[target_column].transform(
                lambda x: x.rolling(window=window * 24, min_periods=1).mean()
            )
            rolling_features[f'价格滚动标准差_{window}天'] = rolling_features.groupby('小时')[target_column].transform(
                lambda x: x.rolling(window=window * 24, min_periods=1).std()
            )
            rolling_features[f'价格滚动最大值_{window}天'] = rolling_features.groupby('小时')[target_column].transform(
                lambda x: x.rolling(window=window * 24, min_periods=1).max()
            )
            rolling_features[f'价格滚动最小值_{window}天'] = rolling_features.groupby('小时')[target_column].transform(
                lambda x: x.rolling(window=window * 24, min_periods=1).min()
            )
            
            # 负荷滚动统计
            if '系统负荷-实时' in rolling_features.columns:
                rolling_features[f'负荷滚动均值_{window}天'] = rolling_features.groupby('小时')['系统负荷-实时'].transform(
                    lambda x: x.rolling(window=window * 24, min_periods=1).mean()
                )
        
        print(f"滚动特征构建完成，共 {len([col for col in rolling_features.columns if '滚动' in col])} 个滚动特征")
        
        return rolling_features
    
    def process_day_ahead_features(self, day_ahead_data: pd.DataFrame) -> pd.DataFrame:
        """
        处理日前披露特征
        
        Parameters
        ----------
        day_ahead_data : pd.DataFrame
            日前披露数据
            
        Returns
        -------
        processed_features : pd.DataFrame
            处理后的日前特征
        """
        if day_ahead_data is None:
            return pd.DataFrame()
        
        processed_features = day_ahead_data.copy()
        
        # 确保数据按小时排序
        processed_features = processed_features.sort_values('小时')
        
        # 日前-实时偏差历史统计（需要历史数据，这里先创建占位符）
        processed_features['日前实时偏差_历史均值'] = 0.0  # 占位符
        processed_features['日前实时偏差_历史标准差'] = 0.0  # 占位符
        
        print(f"日前特征处理完成，共 {len(processed_features.columns)} 个特征")
        
        return processed_features
    
    def create_derived_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建衍生特征
        
        Parameters
        ----------
        features_df : pd.DataFrame
            基础特征数据
            
        Returns
        -------
        derived_features : pd.DataFrame
            衍生特征
        """
        derived_features = features_df.copy()
        
        # 供需差特征
        if all(col in derived_features.columns for col in ['系统负荷-实时', '风电出力-实时', '水电出力-实时', '非市场化机组出力-实时']):
            derived_features['供需差'] = (
                derived_features['系统负荷-实时'] - 
                derived_features['风电出力-实时'] - 
                derived_features['水电出力-实时'] - 
                derived_features['非市场化机组出力-实时']
            )
        
        # 新能源占比
        if all(col in derived_features.columns for col in ['风电出力-实时', '光伏出力-实时', '系统负荷-实时']):
            derived_features['新能源占比'] = (
                (derived_features['风电出力-实时'] + derived_features['光伏出力-实时']) / 
                derived_features['系统负荷-实时']
            ).fillna(0)
        
        # 水电占比
        if all(col in derived_features.columns for col in ['水电出力-实时', '系统负荷-实时']):
            derived_features['水电占比'] = (
                derived_features['水电出力-实时'] / derived_features['系统负荷-实时']
            ).fillna(0)
        
        # 日前-实时偏差（如果同时有日前和实时数据）
        if all(col in derived_features.columns for col in ['系统负荷-日前', '系统负荷-实时']):
            derived_features['负荷预测偏差'] = (
                derived_features['系统负荷-日前'] - derived_features['系统负荷-实时']
            )
        
        # 价格波动特征
        if '价格滚动标准差_7天' in derived_features.columns:
            derived_features['价格波动率'] = derived_features['价格滚动标准差_7天'] / (
                derived_features['价格滚动均值_7天'] + 1e-6
            )
        
        print(f"衍生特征构建完成，共 {len([col for col in derived_features.columns if col not in features_df.columns])} 个衍生特征")
        
        return derived_features
    
    def build_features(self, historical_data: pd.DataFrame, 
                      day_ahead_data: pd.DataFrame, 
                      target_date: pd.Timestamp,
                      include_weather_cluster: bool = False) -> Dict[str, pd.DataFrame]:
        """
        构建完整的特征集
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            历史实时数据（T-2及之前）
        day_ahead_data : pd.DataFrame
            日前披露数据（T日披露T+1）
        target_date : pd.Timestamp
            目标日期（T+1）
        include_weather_cluster : bool
            是否包含天气聚类特征
            
        Returns
        -------
        feature_dict : dict
            包含各类特征的字典
        """
        print("开始构建特征...")
        
        # 1. 构建时间特征
        time_features = self.create_time_features(day_ahead_data, target_date)
        
        # 2. 构建滞后特征
        lag_features = self.create_lag_features(historical_data)
        
        # 3. 构建滚动特征
        rolling_features = self.create_rolling_features(historical_data)
        
        # 4. 处理日前特征
        day_ahead_processed = self.process_day_ahead_features(day_ahead_data)
        
        # 5. 合并基础特征
        # 重置所有DataFrame的索引以确保一致性
        time_features = time_features.reset_index(drop=True)
        lag_features = lag_features.reset_index(drop=True)
        rolling_features = rolling_features.reset_index(drop=True)
        day_ahead_processed = day_ahead_processed.reset_index(drop=True)
        
        # 确保所有DataFrame长度一致
        max_length = max(len(time_features), len(lag_features), len(rolling_features), len(day_ahead_processed))
        
        # 合并特征
        base_features = pd.DataFrame(index=range(max_length))
        
        # 逐个添加特征列，避免重复
        for df, name in [(time_features, 'time'), (lag_features, 'lag'), 
                         (rolling_features, 'rolling'), (day_ahead_processed, 'day_ahead')]:
            for col in df.columns:
                if col not in base_features.columns:
                    if len(df) == max_length:
                        base_features[col] = df[col].values
                    else:
                        # 如果长度不匹配，用NaN填充
                        base_features[col] = np.nan
                        base_features.loc[:len(df)-1, col] = df[col].values
        
        # 6. 构建衍生特征
        all_features = self.create_derived_features(base_features)
        
        # 7. 处理缺失值
        all_features = self.handle_missing_values(all_features)
        
        # 8. 特征选择（可选）
        selected_features = self.select_features(all_features, historical_data[self.feature_config['target_column']])
        
        feature_report = {
            'time_features': time_features,
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'day_ahead_features': day_ahead_processed,
            'derived_features': all_features[[col for col in all_features.columns if col not in base_features.columns]],
            'all_features': all_features,
            'selected_features': selected_features
        }
        
        print(f"特征构建完成，总共 {len(all_features.columns)} 个特征")
        
        return feature_report
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        处理缺失值
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        strategy : str
            缺失值处理策略：'median', 'mean', 'zero', 'drop'
            
        Returns
        -------
        processed_df : pd.DataFrame
            处理后的数据
        """
        processed_df = df.copy()
        
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_count = processed_df[col].isna().sum()
            if missing_count > 0:
                if strategy == 'median':
                    fill_value = processed_df[col].median()
                elif strategy == 'mean':
                    fill_value = processed_df[col].mean()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = processed_df[col].median()  # 默认使用中位数
                
                processed_df[col] = processed_df[col].fillna(fill_value)
                print(f"列 {col} 的 {missing_count} 个缺失值已用 {fill_value:.2f} 填充")
        
        return processed_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', 
                       k: int = 50) -> pd.DataFrame:
        """
        特征选择
        
        Parameters
        ----------
        X : pd.DataFrame
            特征数据
        y : pd.Series
            目标变量
        method : str
            特征选择方法：'mutual_info', 'f_regression', 'variance'
        k : int
            选择前k个特征
            
        Returns
        -------
        selected_features : pd.DataFrame
            选择后的特征
        """
        if len(X.columns) <= k:
            return X  # 如果特征数不多，不进行选择
        
        # 数据预处理：处理无限大和过大数值
        X_clean = X.copy()
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        
        # 处理无限大值
        X_clean[numeric_cols] = X_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 处理过大数值（使用分位数截断）
        for col in numeric_cols:
            q_low = X_clean[col].quantile(0.01)
            q_high = X_clean[col].quantile(0.99)
            X_clean[col] = np.clip(X_clean[col], q_low, q_high)
        
        # 移除常数特征
        variance_threshold = 0.01
        variance = X_clean[numeric_cols].var()
        constant_cols = variance[variance < variance_threshold].index.tolist()
        
        if constant_cols:
            X_clean = X_clean.drop(columns=constant_cols)
            print(f"移除 {len(constant_cols)} 个常数特征")
        
        if len(X_clean.columns) <= k:
            return X_clean
        
        # 特征选择
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            return X_clean  # 不进行基于统计的特征选择
        
        try:
            # 确保y的长度与X匹配
            if len(y) != len(X_clean):
                # 如果长度不匹配，使用前min(len(y), len(X_clean))个样本
                min_len = min(len(y), len(X_clean))
                X_clean = X_clean.iloc[:min_len]
                y = y.iloc[:min_len]
            
            X_selected = selector.fit_transform(X_clean, y)
            selected_cols = X_clean.columns[selector.get_support()]
            
            # 记录特征重要性
            self.feature_importance = dict(zip(selected_cols, selector.scores_))
            
            print(f"特征选择完成，从 {len(X_clean.columns)} 个特征中选择前 {k} 个重要特征")
            
            return X_clean[selected_cols]
        except Exception as e:
            print(f"特征选择失败: {e}，返回原始特征")
            return X_clean
    
    def fit_scaler(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        拟合特征标准化器
        
        Parameters
        ----------
        X : pd.DataFrame
            训练数据
        method : str
            标准化方法：'standard', 'minmax'
            
        Returns
        -------
        X_scaled : pd.DataFrame
            标准化后的数据
        """
        # 数据预处理：处理无限大和缺失值
        X_clean = X.copy()
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        
        # 处理无限大值
        X_clean[numeric_cols] = X_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 处理缺失值（使用中位数填充）
        for col in numeric_cols:
            if X_clean[col].isna().sum() > 0:
                fill_value = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(fill_value)
        
        # 处理过大数值（使用分位数截断）
        for col in numeric_cols:
            q_low = X_clean[col].quantile(0.01)
            q_high = X_clean[col].quantile(0.99)
            X_clean[col] = np.clip(X_clean[col], q_low, q_high)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        X_scaled = X_clean.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X_clean[numeric_cols])
        
        print(f"特征标准化完成，使用 {method} 方法")
        
        return X_scaled
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换特征（使用已拟合的标准化器）
        
        Parameters
        ----------
        X : pd.DataFrame
            待转换数据
            
        Returns
        -------
        X_transformed : pd.DataFrame
            转换后的数据
        """
        if self.scaler is None:
            print("警告: 尚未拟合标准化器，返回原始数据")
            return X
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_transformed = X.copy()
        X_transformed[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X_transformed
    
    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        创建所有特征（简化版，用于批量训练）
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        features_df : pd.DataFrame
            特征数据框
        target_cols : list
            目标列名列表
        """
        print("开始创建所有特征...")
        
        # 复制数据
        features_df = df.copy()
        
        # 确保日期列存在
        if '日期' in features_df.columns:
            features_df['Date'] = pd.to_datetime(features_df['日期'])
        
        # 确保小时列存在
        if '小时' not in features_df.columns:
            features_df['小时'] = features_df.index % 24
        
        # 创建时间特征
        features_df['月份'] = features_df['月'].astype(int) if '月' in features_df.columns else features_df['Date'].dt.month
        features_df['星期'] = features_df['星期'].astype(int) if '星期' in features_df.columns else features_df['Date'].dt.dayofweek + 1
        features_df['是否周末'] = (features_df['星期'] >= 6).astype(int)
        features_df['是否高峰时段'] = ((features_df['小时'] >= 8) & (features_df['小时'] <= 22)).astype(int)
        
        # 创建滞后特征
        target_col = '平均出清价格-实时（元/MWh）'
        if target_col in features_df.columns:
            for lag in [1, 2, 3, 7, 14]:
                features_df[f'{target_col}_lag_{lag}'] = features_df[target_col].shift(lag * 24)
        
        # 创建滚动统计特征
        if target_col in features_df.columns:
            features_df[f'{target_col}_rolling_mean_7d'] = features_df[target_col].shift(24).rolling(window=7*24).mean()
            features_df[f'{target_col}_rolling_std_7d'] = features_df[target_col].shift(24).rolling(window=7*24).std()
        
        # 处理缺失值
        features_df = self.handle_missing_values(features_df, strategy='median')
        
        # 定义目标列（24小时的价格）
        target_cols = []
        for h in range(24):
            col_name = f'Price_H{h:02d}'
            if col_name in features_df.columns:
                target_cols.append(col_name)
        
        # 如果没有预定义的目标列，使用实时价格作为目标
        if len(target_cols) == 0 and target_col in features_df.columns:
            # 创建24小时的目标列
            for h in range(24):
                col_name = f'Price_H{h:02d}'
                hour_mask = features_df['小时'] == h
                features_df.loc[hour_mask, col_name] = features_df.loc[hour_mask, target_col]
                target_cols.append(col_name)
        
        # 处理目标列中的缺失值（使用前向填充+后向填充）
        for col in target_cols:
            if features_df[col].isna().sum() > 0:
                features_df[col] = features_df[col].ffill().bfill()
        
        print(f"特征创建完成，共 {len(features_df.columns)} 个特征，{len(target_cols)} 个目标列")
        
        return features_df, target_cols
    
    def save_features(self, features_df: pd.DataFrame, target_cols: List[str], 
                     feature_path: Optional[str] = None):
        """
        保存特征数据到文件
        
        Parameters
        ----------
        features_df : pd.DataFrame
            特征数据框
        target_cols : list
            目标列名列表
        feature_path : str, optional
            保存路径，默认使用config中的features路径
        """
        if feature_path is None:
            feature_path = config.data_paths['features']
        
        os.makedirs(feature_path, exist_ok=True)
        
        # 保存特征数据
        data_file = os.path.join(feature_path, 'features.csv')
        features_df.to_csv(data_file, index=False, encoding='utf-8-sig')
        logger.info(f"特征数据已保存: {data_file}")
        
        # 保存目标列信息
        import json
        info_file = os.path.join(feature_path, 'feature_info.json')
        feature_info = {
            'target_columns': target_cols,
            'feature_columns': [col for col in features_df.columns if col not in target_cols],
            'n_samples': len(features_df),
            'n_features': len(features_df.columns) - len(target_cols),
            'n_targets': len(target_cols)
        }
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        logger.info(f"特征信息已保存: {info_file}")
    
    def load_features(self, feature_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        从文件加载特征数据
        
        Parameters
        ----------
        feature_path : str, optional
            特征数据路径，默认使用config中的features路径
            
        Returns
        -------
        features_df : pd.DataFrame
            特征数据框
        target_cols : list
            目标列名列表
        """
        if feature_path is None:
            feature_path = config.data_paths['features']
        
        # 加载特征数据
        data_file = os.path.join(feature_path, 'features.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"特征数据文件不存在: {data_file}，请先运行特征工程生成特征")
        
        features_df = pd.read_csv(data_file)
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        logger.info(f"特征数据已加载: {data_file}，共 {len(features_df)} 条记录")
        
        # 加载目标列信息
        import json
        info_file = os.path.join(feature_path, 'feature_info.json')
        with open(info_file, 'r', encoding='utf-8') as f:
            feature_info = json.load(f)
        target_cols = feature_info['target_columns']
        logger.info(f"目标列: {len(target_cols)} 个，特征列: {len(feature_info['feature_columns'])} 个")
        
        return features_df, target_cols
    
    def generate_feature_dataset(self, data_loader: 'DataLoader', 
                               start_date: str = None,
                               end_date: str = None,
                               save_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        生成完整的特征数据集
        
        Parameters
        ----------
        data_loader : DataLoader
            数据加载器实例
        start_date : str, optional
            开始日期，默认使用配置中的开始日期
        end_date : str, optional
            结束日期，默认使用配置中的结束日期
        save_path : str, optional
            保存路径，默认使用配置中的特征路径
            
        Returns
        -------
        dataset_dict : dict
            包含特征数据集的字典
        """
        if start_date is None:
            start_date = config.data_config['start_date']
        if end_date is None:
            end_date = config.data_config['end_date']
        if save_path is None:
            save_path = config.data_paths['features']
        
        # 确保保存路径存在
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        print(f"开始生成特征数据集: {start_date} 至 {end_date}")
        
        # 加载完整数据
        df = data_loader.load_data()
        
        # 按时间顺序生成特征
        all_features_list = []
        all_targets_list = []
        date_list = []
        
        # 获取所有可预测的日期（T+1日期）
        prediction_dates = pd.date_range(
            start=pd.to_datetime(start_date) + pd.Timedelta(days=3),  # 需要至少3天历史数据
            end=pd.to_datetime(end_date),
            freq='D'
        )
        
        print(f"需要生成 {len(prediction_dates)} 天的特征")
        
        for i, prediction_date in enumerate(prediction_dates):
            if i % 30 == 0:  # 每30天显示一次进度
                print(f"进度: {i}/{len(prediction_dates)} - {prediction_date.date()}")
            
            try:
                # 准备预测数据
                prediction_data = data_loader.prepare_prediction_data(df, prediction_date)
                
                # 构建特征
                feature_report = self.build_features(
                    historical_data=prediction_data['historical_data'],
                    day_ahead_data=prediction_data['day_ahead_data'],
                    target_date=prediction_date
                )
                
                # 提取特征和目标
                features = feature_report['all_features']
                target_col = self.feature_config['target_column']
                
                # 确保特征和目标对齐
                if len(features) == 24:  # 24小时数据
                    # 获取对应的目标值（T+1日的实时价格）
                    target_data = df[
                        (df['日期'] == prediction_date) & 
                        (df[target_col].notna())
                    ][['小时', target_col]].sort_values('小时')
                    
                    if len(target_data) == 24:
                        # 添加日期和小时信息
                        features['日期'] = prediction_date
                        features['小时'] = range(24)
                        
                        # 添加目标值
                        features[target_col] = target_data[target_col].values
                        
                        all_features_list.append(features)
                        date_list.append(prediction_date)
                
            except Exception as e:
                print(f"日期 {prediction_date} 的特征生成失败: {e}")
                continue
        
        if not all_features_list:
            raise ValueError("未能生成任何有效的特征数据")
        
        # 合并所有特征
        full_dataset = pd.concat(all_features_list, ignore_index=True)
        
        # 数据集划分
        dataset_dict = self.split_dataset_by_time(full_dataset)
        
        # 保存数据集
        if save_path:
            self.save_feature_dataset(dataset_dict, save_path)
        
        print(f"特征数据集生成完成，共 {len(full_dataset)} 条记录")
        
        return dataset_dict
    
    def split_dataset_by_time(self, dataset: pd.DataFrame, 
                            test_start_date: str = None,
                            val_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        按时间顺序划分数据集
        
        Parameters
        ----------
        dataset : pd.DataFrame
            完整数据集
        test_start_date : str, optional
            测试集开始日期，默认使用配置
        val_ratio : float
            验证集比例（从训练集中划分）
            
        Returns
        -------
        dataset_dict : dict
            划分后的数据集
        """
        if test_start_date is None:
            test_start_date = config.data_config['test_start_date']
        
        test_start = pd.to_datetime(test_start_date)
        
        # 划分测试集
        test_mask = dataset['日期'] >= test_start
        test_set = dataset[test_mask].copy()
        train_val_set = dataset[~test_mask].copy()
        
        # 从训练集中划分验证集
        train_val_dates = train_val_set['日期'].unique()
        n_val_dates = max(1, int(len(train_val_dates) * val_ratio))
        val_dates = train_val_dates[-n_val_dates:]
        
        val_mask = train_val_set['日期'].isin(val_dates)
        val_set = train_val_set[val_mask].copy()
        train_set = train_val_set[~val_mask].copy()
        
        # 统计信息
        print(f"数据集划分完成:")
        print(f"  训练集: {train_set['日期'].min().date()} 至 {train_set['日期'].max().date()}, {len(train_set)} 条")
        print(f"  验证集: {val_set['日期'].min().date()} 至 {val_set['日期'].max().date()}, {len(val_set)} 条")
        print(f"  测试集: {test_set['日期'].min().date()} 至 {test_set['日期'].max().date()}, {len(test_set)} 条")
        
        return {
            'train': train_set,
            'val': val_set,
            'test': test_set,
            'full': dataset
        }
    
    def save_feature_dataset(self, dataset_dict: Dict[str, pd.DataFrame], 
                           save_path: str) -> None:
        """
        保存特征数据集
        
        Parameters
        ----------
        dataset_dict : dict
            数据集字典
        save_path : str
            保存路径
        """
        # 保存各个数据集
        for dataset_name, dataset in dataset_dict.items():
            file_path = Path(save_path) / f"features_{dataset_name}.csv"
            dataset.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"保存 {dataset_name} 数据集: {file_path}")
        
        # 保存特征信息
        feature_info = {
            'feature_columns': list(dataset_dict['train'].columns),
            'target_column': self.feature_config['target_column'],
            'dataset_sizes': {k: len(v) for k, v in dataset_dict.items()},
            'date_ranges': {k: (v['日期'].min().date(), v['日期'].max().date()) 
                          for k, v in dataset_dict.items() if len(v) > 0}
        }
        
        import json
        info_path = Path(save_path) / "feature_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"保存特征信息: {info_path}")
    
    def load_feature_dataset(self, load_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        加载特征数据集
        
        Parameters
        ----------
        load_path : str, optional
            加载路径，默认使用配置中的特征路径
            
        Returns
        -------
        dataset_dict : dict
            加载的数据集
        """
        if load_path is None:
            load_path = config.data_paths['features']
        
        dataset_dict = {}
        
        for dataset_name in ['train', 'val', 'test', 'full']:
            file_path = Path(load_path) / f"features_{dataset_name}.csv"
            if file_path.exists():
                dataset = pd.read_csv(file_path)
                # 转换日期列
                if '日期' in dataset.columns:
                    dataset['日期'] = pd.to_datetime(dataset['日期'])
                dataset_dict[dataset_name] = dataset
                print(f"加载 {dataset_name} 数据集: {len(dataset)} 条记录")
            else:
                print(f"警告: 未找到 {file_path}")
        
        return dataset_dict
    
    def prepare_training_data(self, dataset_dict: Dict[str, pd.DataFrame],
                            scale_features: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        准备训练数据（特征和目标分离）
        
        Parameters
        ----------
        dataset_dict : dict
            数据集字典
        scale_features : bool
            是否对特征进行标准化
            
        Returns
        -------
        training_data : dict
            准备好的训练数据
        """
        training_data = {}
        target_col = self.feature_config['target_column']
        
        for dataset_name, dataset in dataset_dict.items():
            if len(dataset) == 0:
                continue
            
            # 分离特征和目标
            feature_cols = [col for col in dataset.columns if col != target_col and col != '日期']
            X = dataset[feature_cols]
            y = dataset[target_col]
            
            # 特征标准化
            if scale_features and len(X) > 0:
                if self.scaler is None:
                    X_scaled = self.fit_scaler(X)
                else:
                    X_scaled = self.transform_features(X)
            else:
                X_scaled = X
            
            training_data[dataset_name] = (X_scaled, y)
            
            print(f"准备 {dataset_name} 数据: {X_scaled.shape} 特征, {len(y)} 目标")
        
        return training_data


def test_basic_features():
    """
    测试基础特征构建功能
    """
    print("=== 基础特征构建测试 ===")
    
    # 初始化特征工程器
    engineer = FeatureEngineer()
    
    # 测试数据（模拟）
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    test_data = pd.DataFrame({
        '日期': dates.date,
        '小时': dates.hour,
        '星期': dates.dayofweek,
        '月': dates.month,
        '系统负荷-实时': np.random.normal(20000, 5000, 100),
        '平均出清价格-实时（元/MWh）': np.random.normal(400, 100, 100),
        '风电出力-实时': np.random.normal(3000, 1000, 100),
        '水电出力-实时': np.random.normal(5000, 1000, 100)
    })
    
    # 测试时间特征
    time_features = engineer.create_time_features(test_data.head(24), pd.Timestamp('2024-01-02'))
    print(f"时间特征测试: {len(time_features.columns)} 个特征")
    
    # 测试滞后特征
    lag_features = engineer.create_lag_features(test_data)
    print(f"滞后特征测试: {len([col for col in lag_features.columns if '滞后' in col])} 个滞后特征")
    
    # 测试滚动特征
    rolling_features = engineer.create_rolling_features(test_data)
    print(f"滚动特征测试: {len([col for col in rolling_features.columns if '滚动' in col])} 个滚动特征")
    
    # 测试衍生特征
    derived_features = engineer.create_derived_features(test_data)
    print(f"衍生特征测试: {len([col for col in derived_features.columns if col not in test_data.columns])} 个衍生特征")
    
    print("\n=== 基础特征构建测试完成 ===")


def test_feature_dataset_generation():
    """
    测试特征数据集生成功能
    """
    print("\n=== 特征数据集生成测试 ===")
    
    # 初始化数据加载器和特征工程器
    from data_loader import DataLoader
    data_loader = DataLoader()
    engineer = FeatureEngineer()
    
    try:
        # 测试小规模数据集生成（仅生成几天的数据用于测试）
        test_start_date = '2025-03-01'
        test_end_date = '2025-03-05'  # 只生成5天的数据用于测试
        
        print(f"测试数据集生成: {test_start_date} 至 {test_end_date}")
        
        # 生成特征数据集
        dataset_dict = engineer.generate_feature_dataset(
            data_loader=data_loader,
            start_date=test_start_date,
            end_date=test_end_date,
            save_path=config.data_paths['features'] / 'test'
        )
        
        # 测试数据集加载
        print("\n=== 测试数据集加载 ===")
        loaded_dataset = engineer.load_feature_dataset(
            load_path=config.data_paths['features'] / 'test'
        )
        
        # 测试训练数据准备
        print("\n=== 测试训练数据准备 ===")
        training_data = engineer.prepare_training_data(loaded_dataset)
        
        for dataset_name, (X, y) in training_data.items():
            if len(X) > 0:
                print(f"{dataset_name}集: {X.shape} 特征, {len(y)} 目标")
        
        print("\n=== 特征数据集生成测试完成 ===")
        
    except Exception as e:
        print(f"特征数据集生成测试失败: {e}")
        print("这可能是因为测试数据范围太小或数据不完整")


def main():
    """
    特征工程模块主测试函数
    """
    print("=== 特征工程模块综合测试 ===")
    
    # 测试基础特征构建
    test_basic_features()
    
    print("\n=== 特征工程模块测试完成 ===")


if __name__ == '__main__':
    main()