"""
滑动窗口Direct多步预测特征工程

预测场景：T时刻预测T+1日的24小时实时价格
特征设计：
- 实时价格滞后：仅使用T-2日（避免数据泄露）
- 其他特征：当前时点的1h滞后 + 1h未来（最后1个时点无未来）
- 预测策略：Direct策略，24个独立模型
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


class DirectFeatureEngineer:
    """
    Direct多步预测特征工程类
    
    为每个预测步长（0-23时）构建滑动窗口特征
    """
    
    def __init__(self):
        self.price_col = '平均出清价格-实时（元/MWh）'
        self.market_cols = [
            '系统负荷-实时', '非市场化机组出力-实时', '风电出力-实时',
            '光伏出力-实时', '水电出力-实时', '联络线计划-实时'
        ]
    
    def create_all_features(self, df: pd.DataFrame) -> Dict[int, Tuple[pd.DataFrame, str]]:
        """
        为每个预测步长创建滑动窗口特征
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据（processed_data.csv）
            
        Returns
        -------
        hourly_features : dict
            {hour: (features_df, target_col)} 每个预测步长的特征数据
        """
        print("开始创建Direct多步预测特征...")
        
        # 数据准备
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(['日期', '小时']).reset_index(drop=True)
        
        all_dates = df['日期'].unique()
        print(f"数据时间范围: {all_dates.min()} 至 {all_dates.max()}")
        print(f"总天数: {len(all_dates)}")
        
        # 为每个预测步长创建特征
        hourly_results = {}
        
        for target_hour in range(24):
            print(f"\n创建预测T+1日 {target_hour:02d}:00 的特征...")
            features_list = []
            
            # 从第3天开始（需要T-2的历史数据）
            for i in range(2, len(all_dates)):
                target_date = all_dates[i]  # T+1
                
                # 获取T-2日数据（实时价格滞后）
                t_minus_2_date = all_dates[i-2]
                t_minus_2_data = df[df['日期'] == t_minus_2_date].copy()
                if len(t_minus_2_data) != 24:
                    continue
                
                # 获取T+1日数据（其他特征）
                target_date_data = df[df['日期'] == target_date].copy()
                if len(target_date_data) != 24:
                    continue
                
                # 构建该预测步长的滑动窗口特征
                sample_features = self._build_direct_features(
                    t_minus_2_data=t_minus_2_data,
                    target_date_data=target_date_data,
                    target_date=target_date,
                    target_hour=target_hour
                )
                
                if sample_features is not None:
                    features_list.append(sample_features)
                
                if (i - 2) % 50 == 0 and (i - 2) > 0:
                    print(f"  进度: {i-2}/{len(all_dates)-2}")
            
            if features_list:
                features_df = pd.DataFrame(features_list)
                features_df = self._handle_missing_values(features_df)
                target_col = f'Price_H{target_hour:02d}'
                hourly_results[target_hour] = (features_df, target_col)
                print(f"  完成: {len(features_df)} 样本, {len(features_df.columns)-2} 特征")
            else:
                print(f"  警告: {target_hour:02d}:00 未能生成特征")
        
        print(f"\nDirect多步预测特征创建完成:")
        print(f"  共 {len(hourly_results)} 个预测步长")
        
        return hourly_results
    
    def _build_direct_features(self, t_minus_2_data: pd.DataFrame,
                               target_date_data: pd.DataFrame,
                               target_date: pd.Timestamp,
                               target_hour: int) -> Optional[Dict]:
        """
        构建单个预测步长的滑动窗口特征
        
        Parameters
        ----------
        t_minus_2_data : pd.DataFrame
            T-2日数据（用于实时价格滞后）
        target_date_data : pd.DataFrame
            T+1日数据（用于其他特征）
        target_date : pd.Timestamp
            目标日期（T+1）
        target_hour : int
            目标小时（预测步长）
            
        Returns
        -------
        features : dict or None
            特征字典
        """
        features = {}
        
        # 1. 时间特征
        features['预测日期'] = target_date
        features['月份'] = target_date.month
        features['星期'] = target_date.dayofweek + 1
        features['是否周末'] = 1 if target_date.dayofweek >= 5 else 0
        features['季度'] = (target_date.month - 1) // 3 + 1
        features['目标小时'] = target_hour
        
        # 2. 实时价格滞后特征（仅使用T-2日同一时刻）
        # 这是唯一可以使用历史实时价格的地方
        t_minus_2_hour_data = t_minus_2_data[t_minus_2_data['小时'] == target_hour]
        if len(t_minus_2_hour_data) == 1:
            lag_price = t_minus_2_hour_data.iloc[0][self.price_col]
            if pd.notna(lag_price):
                features[f'滞后2天_H{target_hour:02d}_价格'] = lag_price
        
        # 3. 其他特征的滑动窗口（日前市场 + 气象）
        # 对于T+1日target_hour时刻，使用1h滞后和1h未来
        # 注意：所有时刻都使用T+1日的日前数据，所以00:00也可以使用23:00（前一天）的数据
        
        # 3.1 当前时刻特征
        self._add_hour_features(features, target_date_data, target_hour, '当前')
        
        # 3.2 1h滞后特征
        # 00:00的滞后1h是T+1日的23:00（前一天的23点，但属于T+1日的日前数据）
        lag_hour = target_hour - 1 if target_hour > 0 else 23
        self._add_hour_features(features, target_date_data, lag_hour, '滞后1h')
        
        # 3.3 1h未来特征
        # 23:00的未来1h是T+1日的00:00（次日的0点，但属于T+1日的日前数据）
        future_hour = target_hour + 1 if target_hour < 23 else 0
        self._add_hour_features(features, target_date_data, future_hour, '未来1h')
        
        # 4. 目标值（T+1日target_hour时刻的实时价格）
        target_hour_data = target_date_data[target_date_data['小时'] == target_hour]
        if len(target_hour_data) == 1:
            target_price = target_hour_data.iloc[0][self.price_col]
            if pd.notna(target_price):
                features[f'Price_H{target_hour:02d}'] = target_price
            else:
                return None
        else:
            return None
        
        return features
    
    def _add_hour_features(self, features: Dict, date_data: pd.DataFrame, 
                          hour: int, prefix: str):
        """
        添加指定时刻的特征（日前市场 + 气象）
        
        Parameters
        ----------
        features : dict
            特征字典
        date_data : pd.DataFrame
            日期数据
        hour : int
            小时
        prefix : str
            特征前缀（'当前', '滞后1h', '未来1h'）
        """
        hour_data = date_data[date_data['小时'] == hour]
        if len(hour_data) != 1:
            return
        
        row = hour_data.iloc[0]
        
        # 日前市场特征
        for col in self.market_cols:
            if col in row.index and pd.notna(row[col]):
                short_name = col.replace('（元/MWh）', '').replace('-实时', '')
                features[f'{prefix}_日前_{short_name}'] = row[col]
        
        # 气象特征
        weather_cols = [c for c in date_data.columns 
                       if any(x in c for x in ['温度', '风速', '湿度', '压强', '云量', '辐照度', '降雨量'])]
        for col in weather_cols:
            if col in row.index and pd.notna(row[col]):
                short_name = col.replace('-预测', '').replace('-实际', '')
                features[f'{prefix}_气象_{short_name}'] = row[col]
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def save_features(self, hourly_results: Dict[int, Tuple[pd.DataFrame, str]],
                     feature_path: Optional[Path] = None):
        """保存特征数据"""
        if feature_path is None:
            feature_path = config.get_data_path('features') / 'direct'
        
        os.makedirs(feature_path, exist_ok=True)
        
        # 保存每个预测步长的特征
        for hour, (features_df, target_col) in hourly_results.items():
            data_file = feature_path / f'features_H{hour:02d}.csv'
            features_df.to_csv(data_file, index=False, encoding='utf-8-sig')
        
        # 保存特征信息
        import json
        info_file = feature_path / 'feature_info.json'
        
        first_hour = list(hourly_results.keys())[0]
        first_df, _ = hourly_results[first_hour]
        
        # 统计各时刻特征数
        feature_counts = {h: len(df.columns) - 2 for h, (df, _) in hourly_results.items()}
        
        feature_info = {
            'type': 'direct_multi_step',
            'n_hours': len(hourly_results),
            'hours': list(hourly_results.keys()),
            'n_samples_per_hour': len(first_df),
            'feature_counts_per_hour': feature_counts,
            'description': '滑动窗口Direct多步预测，实时价格仅T-2，其他特征1h滞后+1h未来'
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Direct多步预测特征已保存到: {feature_path}")
        logger.info(f"特征数统计: {feature_counts}")
    
    def load_features(self, hour: int, feature_path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
        """加载指定预测步长的特征"""
        if feature_path is None:
            feature_path = config.get_data_path('features') / 'direct'
        
        data_file = feature_path / f'features_H{hour:02d}.csv'
        features_df = pd.read_csv(data_file)
        
        if '预测日期' in features_df.columns:
            features_df['预测日期'] = pd.to_datetime(features_df['预测日期'])
        
        target_col = f'Price_H{hour:02d}'
        
        return features_df, target_col


class DirectMultiStepModel:
    """
    Direct多步预测模型包装器
    
    包装24个独立模型，提供统一的fit/predict接口
    """
    
    def __init__(self, model_type: str = 'lightgbm', **model_params):
        """
        初始化Direct多步预测模型
        
        Parameters
        ----------
        model_type : str
            基模型类型
        **model_params : dict
            模型参数
        """
        self.model_type = model_type
        self.model_params = model_params
        self.models = {}  # {hour: model}
        self.is_fitted = False
        
    def fit(self, hourly_data: Dict[int, Tuple[pd.DataFrame, str]]):
        """
        训练24个独立模型
        
        Parameters
        ----------
        hourly_data : dict
            {hour: (features_df, target_col)}
        """
        from sklearn.linear_model import Ridge, Lasso
        
        for hour in range(24):
            if hour not in hourly_data:
                continue
            
            features_df, target_col = hourly_data[hour]
            feature_cols = [c for c in features_df.columns 
                          if c not in [target_col, '预测日期']]
            
            X = features_df[feature_cols].values
            y = features_df[target_col].values
            
            # 创建并训练模型
            if self.model_type == 'lightgbm':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(**self.model_params, verbose=-1)
            elif self.model_type == 'xgboost':
                from xgboost import XGBRegressor
                model = XGBRegressor(**self.model_params)
            elif self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**self.model_params)
            elif self.model_type == 'ridge':
                model = Ridge(**self.model_params)
            elif self.model_type == 'lasso':
                model = Lasso(**self.model_params)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            model.fit(X, y)
            self.models[hour] = model
            
        self.is_fitted = True
        return self
    
    def predict(self, X_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """
        多步预测
        
        Parameters
        ----------
        X_dict : dict
            {hour: X_features} 每个预测步长的特征
            
        Returns
        -------
        predictions : np.ndarray
            形状为 (n_samples, 24) 的预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = []
        for hour in range(24):
            if hour in self.models and hour in X_dict:
                pred = self.models[hour].predict(X_dict[hour])
                predictions.append(pred)
            else:
                raise ValueError(f"缺少 {hour}:00 的模型或特征")
        
        return np.column_stack(predictions)
    
    def save(self, save_dir: Path):
        """保存所有模型"""
        import joblib
        os.makedirs(save_dir, exist_ok=True)
        
        for hour, model in self.models.items():
            joblib.dump(model, save_dir / f'model_H{hour:02d}.pkl')
        
        # 保存配置
        config = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'hours': list(self.models.keys())
        }
        joblib.dump(config, save_dir / 'config.pkl')
    
    @classmethod
    def load(cls, save_dir: Path):
        """加载所有模型"""
        import joblib
        
        config = joblib.load(save_dir / 'config.pkl')
        instance = cls(config['model_type'], **config['model_params'])
        
        for hour in config['hours']:
            instance.models[hour] = joblib.load(save_dir / f'model_H{hour:02d}.pkl')
        
        instance.is_fitted = True
        return instance


def main():
    """主函数"""
    print("=" * 60)
    print("开始生成Direct多步预测特征")
    print("=" * 60)
    
    processed_file = config.get_data_path('processed_data')
    if not processed_file.exists():
        logger.error(f"预处理数据不存在: {processed_file}")
        return
    
    df = pd.read_csv(processed_file)
    logger.info(f"数据加载完成: {df.shape}")
    
    engineer = DirectFeatureEngineer()
    hourly_results = engineer.create_all_features(df)
    engineer.save_features(hourly_results)
    
    print("\n" + "=" * 60)
    print("Direct多步预测特征生成完成")
    print("=" * 60)
    
    # 打印特征示例
    if hourly_results:
        for hour in [0, 8, 12, 23]:
            if hour in hourly_results:
                df_h, target_col = hourly_results[hour]
                feature_cols = [c for c in df_h.columns if c not in [target_col, '预测日期']]
                print(f"\n{hour:02d}:00时刻 - 特征数: {len(feature_cols)}")
                print(f"  特征示例: {feature_cols[:8]}")


if __name__ == '__main__':
    main()
