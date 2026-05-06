"""
滑动窗口Direct多步预测特征工程

预测场景：T时刻预测T+1日的24小时实时价格
特征设计：
- 实时价格滞后：仅使用T-2及之前的实时价格（避免数据泄露）
- 可用市场边界：当前时点、1h滞后、1h未来及其衍生净负荷/爬坡特征
- 气象特征：当前时点、1h滞后、1h未来
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
        self.price_lags = [2, 3, 4, 7, 14]
        self.price_windows = [3, 7, 14]
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
        daily_data = {date: group.copy() for date, group in df.groupby('日期', sort=False)}
        print(f"数据时间范围: {all_dates.min()} 至 {all_dates.max()}")
        print(f"总天数: {len(all_dates)}")
        
        # 为每个预测步长创建特征
        hourly_results = {}
        
        for target_hour in range(24):
            print(f"\n创建预测T+1日 {target_hour:02d}:00 的特征...")
            features_list = []
            
            # 从第16天开始，确保T-2到T-15历史价格窗口完整可用
            min_history_lag = max(max(self.price_lags), max(self.price_windows) + 1)
            for i in range(min_history_lag, len(all_dates)):
                target_date = all_dates[i]  # T+1
                
                # 获取T-2日数据（实时价格滞后）
                t_minus_2_date = all_dates[i-2]
                t_minus_2_data = daily_data.get(t_minus_2_date)
                if t_minus_2_data is None:
                    continue
                if len(t_minus_2_data) != 24:
                    continue
                
                # 获取T+1日数据（其他特征）
                target_date_data = daily_data.get(target_date)
                if target_date_data is None:
                    continue
                if len(target_date_data) != 24:
                    continue
                
                # 构建该预测步长的滑动窗口特征
                sample_features = self._build_direct_features(
                    daily_data=daily_data,
                    all_dates=all_dates,
                    date_index=i,
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
    
    def _build_direct_features(self, daily_data: Dict[pd.Timestamp, pd.DataFrame],
                               all_dates: np.ndarray,
                               date_index: int,
                               t_minus_2_data: pd.DataFrame,
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
        for weekday in range(1, 8):
            features[f'星期_{weekday}'] = 1 if target_date.dayofweek + 1 == weekday else 0
        features['是否周三'] = 1 if target_date.dayofweek == 2 else 0
        features['是否周六'] = 1 if target_date.dayofweek == 5 else 0
        features['是否周日'] = 1 if target_date.dayofweek == 6 else 0
        
        # 2. 历史实时价格特征（仅使用T-2及之前，避免数据泄露）
        self._add_price_history_features(
            features=features,
            daily_data=daily_data,
            all_dates=all_dates,
            date_index=date_index,
            target_hour=target_hour
        )
        self._add_same_weekday_price_features(features, daily_data, all_dates, date_index, target_hour)
        self._add_midday_low_price_features(features, daily_data, all_dates, date_index, target_hour)
        
        # 3. 其他特征的滑动窗口（可用市场边界 + 气象）
        # 对于T+1日target_hour时刻，使用1h滞后和1h未来
        # 注意：这里不使用日前价格；市场边界列来自预处理后的可用输入。
        
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
        self._add_market_change_features(features)
        self._add_market_daily_shape_features(features, target_date_data, target_hour)
        self._add_midday_market_shape_features(features, target_date_data, target_hour)
        self._add_midday_weather_agg_features(features, target_date_data, target_hour)
        
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

    def _add_price_history_features(self, features: Dict,
                                    daily_data: Dict[pd.Timestamp, pd.DataFrame],
                                    all_dates: np.ndarray, date_index: int,
                                    target_hour: int):
        """添加T-2及之前的实时价格滞后和统计特征。"""
        lag_values = {}

        for lag in self.price_lags:
            lag_date = all_dates[date_index - lag]
            lag_data = daily_data.get(lag_date)
            if lag_data is None:
                continue
            price = self._get_hour_price(lag_data, target_hour)
            if price is not None:
                lag_values[lag] = price
                features[f'滞后{lag}天_H{target_hour:02d}_价格'] = price

        # T-2日邻近小时价格，用于捕捉日内曲线形态
        t_minus_2_data = daily_data.get(all_dates[date_index - 2])
        if t_minus_2_data is None:
            return
        for offset, name in [(-1, '前1h'), (1, '后1h')]:
            neighbor_hour = (target_hour + offset) % 24
            price = self._get_hour_price(t_minus_2_data, neighbor_hour)
            if price is not None:
                features[f'滞后2天_H{target_hour:02d}_{name}_价格'] = price

        # T-2日全日形态统计
        day_prices = t_minus_2_data.sort_values('小时')[self.price_col].dropna()
        current_price = lag_values.get(2)
        if len(day_prices) == 24 and current_price is not None:
            day_mean = float(day_prices.mean())
            day_std = float(day_prices.std())
            day_min = float(day_prices.min())
            day_max = float(day_prices.max())
            features['历史价格_T2日均值'] = day_mean
            features['历史价格_T2日标准差'] = day_std
            features['历史价格_T2日最小值'] = day_min
            features['历史价格_T2日最大值'] = day_max
            features[f'历史价格_H{target_hour:02d}_T2相对日均值'] = current_price - day_mean
            features[f'历史价格_H{target_hour:02d}_T2日内极差位置'] = (
                (current_price - day_min) / (day_max - day_min) if day_max != day_min else 0.0
            )
            features[f'历史价格_H{target_hour:02d}_T2低价标记'] = 1 if current_price <= 80 else 0

        # 同小时滚动统计：窗口均从T-2往前取，确保预测时可用
        for window in self.price_windows:
            prices = []
            for lag in range(2, window + 2):
                lag_date = all_dates[date_index - lag]
                lag_data = daily_data.get(lag_date)
                if lag_data is None:
                    continue
                price = self._get_hour_price(lag_data, target_hour)
                if price is not None:
                    prices.append(price)

            if prices:
                values = np.asarray(prices, dtype=float)
                prefix = f'历史价格_H{target_hour:02d}_近{window}日'
                features[f'{prefix}_均值'] = float(values.mean())
                features[f'{prefix}_标准差'] = float(values.std())
                features[f'{prefix}_最小值'] = float(values.min())
                features[f'{prefix}_最大值'] = float(values.max())
                features[f'{prefix}_低价次数'] = int(np.sum(values <= 80))

        if 2 in lag_values and 3 in lag_values:
            features[f'历史价格_H{target_hour:02d}_T2减T3'] = lag_values[2] - lag_values[3]
        if 2 in lag_values and 7 in lag_values:
            features[f'历史价格_H{target_hour:02d}_T2减T7'] = lag_values[2] - lag_values[7]

    def _get_hour_price(self, date_data: pd.DataFrame, hour: int) -> Optional[float]:
        hour_data = date_data[date_data['小时'] == hour]
        if len(hour_data) != 1:
            return None
        price = hour_data.iloc[0][self.price_col]
        if pd.isna(price):
            return None
        return float(price)

    def _add_same_weekday_price_features(self, features: Dict,
                                         daily_data: Dict[pd.Timestamp, pd.DataFrame],
                                         all_dates: np.ndarray, date_index: int,
                                         target_hour: int) -> None:
        """添加同星期历史价格形态，窗口均来自T-7/T-14/T-21。"""
        values = {}
        for lag in [7, 14, 21]:
            if date_index - lag < 0:
                continue
            lag_data = daily_data.get(all_dates[date_index - lag])
            if lag_data is None:
                continue
            price = self._get_hour_price(lag_data, target_hour)
            if price is None:
                continue
            values[lag] = price
            features[f'同星期历史_滞后{lag}天_H{target_hour:02d}_价格'] = price

        if values:
            arr = np.asarray(list(values.values()), dtype=float)
            prefix = f'同星期历史_H{target_hour:02d}_近3周'
            features[f'{prefix}_均值'] = float(arr.mean())
            features[f'{prefix}_最小值'] = float(arr.min())
            features[f'{prefix}_最大值'] = float(arr.max())
            features[f'{prefix}_低价次数'] = int(np.sum(arr <= 80))
        if 7 in values and 14 in values:
            features[f'同星期历史_H{target_hour:02d}_T7减T14'] = values[7] - values[14]

    def _add_midday_low_price_features(self, features: Dict,
                                       daily_data: Dict[pd.Timestamp, pd.DataFrame],
                                       all_dates: np.ndarray, date_index: int,
                                       target_hour: int) -> None:
        """添加T-2日午间低价形态，帮助H08-H15识别低价/近零价风险。"""
        if target_hour < 8 or target_hour > 15:
            return
        t_minus_2_data = daily_data.get(all_dates[date_index - 2])
        if t_minus_2_data is None:
            return

        hour_prices = []
        for hour in range(8, 16):
            price = self._get_hour_price(t_minus_2_data, hour)
            if price is not None:
                hour_prices.append((hour, price))
        if len(hour_prices) != 8:
            return

        hours = np.asarray([item[0] for item in hour_prices], dtype=int)
        prices = np.asarray([item[1] for item in hour_prices], dtype=float)
        current_price = prices[np.where(hours == target_hour)[0][0]]
        midday_mean = float(prices.mean())
        midday_min = float(prices.min())
        midday_max = float(prices.max())
        valley_hour = int(hours[int(np.argmin(prices))])

        features['午间低价_T2_H08H15均值'] = midday_mean
        features['午间低价_T2_H08H15最小值'] = midday_min
        features['午间低价_T2_H08H15最大值'] = midday_max
        features['午间低价_T2_H08H15低价小时数'] = int(np.sum(prices <= 80))
        features['午间低价_T2_H08H15近零小时数'] = int(np.sum(prices <= 20))
        features['午间低价_T2_谷底小时'] = valley_hour
        features['午间低价_T2_目标小时距谷底'] = int(target_hour - valley_hour)
        features[f'午间低价_H{target_hour:02d}_T2相对午间均值'] = float(current_price - midday_mean)
        features[f'午间低价_H{target_hour:02d}_T2日内位置'] = (
            float((current_price - midday_min) / (midday_max - midday_min)) if midday_max != midday_min else 0.0
        )
    
    def _add_hour_features(self, features: Dict, date_data: pd.DataFrame, 
                          hour: int, prefix: str):
        """
        添加指定时刻的特征（可用市场边界 + 气象）
        
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
        
        # 市场边界特征：原始列 + 电价机理相关衍生量，不包含日前价格。
        for name, value in self._compute_market_values(row).items():
            if pd.notna(value):
                features[f'{prefix}_市场_{name}'] = value
        
        # 气象特征
        weather_cols = [c for c in date_data.columns 
                       if any(x in c for x in ['温度', '风速', '湿度', '压强', '云量', '辐照度', '降雨量'])]
        for col in weather_cols:
            if col in row.index and pd.notna(row[col]):
                short_name = col.replace('-预测', '').replace('-实际', '')
                features[f'{prefix}_气象_{short_name}'] = row[col]

    def _compute_market_values(self, row: pd.Series) -> Dict[str, float]:
        """从单小时市场边界中构造电价相关衍生变量。"""
        values = {}

        for col in self.market_cols:
            if col in row.index and pd.notna(row[col]):
                short_name = col.replace('（元/MWh）', '').replace('-实时', '')
                values[short_name] = float(row[col])

        load = values.get('系统负荷')
        wind = values.get('风电出力')
        solar = values.get('光伏出力')
        hydro = values.get('水电出力')
        tie = values.get('联络线计划')
        non_market = values.get('非市场化机组出力')

        renewable = self._sum_available(wind, solar)
        stable_supply = self._sum_available(hydro, non_market, tie)
        covered_supply = self._sum_available(renewable, hydro, non_market, tie)

        if renewable is not None:
            values['新能源出力'] = renewable
        if load is not None and renewable is not None:
            values['净负荷'] = load - renewable
            values['新能源占负荷比'] = self._safe_div(renewable, load)
        if load is not None and renewable is not None and hydro is not None:
            values['剩余负荷'] = load - renewable - hydro
        if load is not None and covered_supply is not None:
            values['市场化缺口'] = load - covered_supply
            values['供需覆盖率'] = self._safe_div(covered_supply, load)

        ratio_inputs = {
            '风电占负荷比': wind,
            '光伏占负荷比': solar,
            '水电占负荷比': hydro,
            '联络线占负荷比': tie,
            '非市场占负荷比': non_market,
            '稳定供给占负荷比': stable_supply,
        }
        if load is not None:
            for name, numerator in ratio_inputs.items():
                if numerator is not None:
                    values[name] = self._safe_div(numerator, load)

        return values

    def _add_market_change_features(self, features: Dict) -> None:
        """添加相邻小时市场边界变化，帮助模型识别爬坡与供需转折。"""
        names = [
            '系统负荷', '风电出力', '光伏出力', '水电出力', '联络线计划', '非市场化机组出力',
            '新能源出力', '净负荷', '剩余负荷', '市场化缺口', '新能源占负荷比', '供需覆盖率'
        ]

        for name in names:
            current_key = f'当前_市场_{name}'
            lag_key = f'滞后1h_市场_{name}'
            future_key = f'未来1h_市场_{name}'

            if current_key in features and lag_key in features:
                features[f'市场变化_当前减滞后1h_{name}'] = features[current_key] - features[lag_key]
            if future_key in features and current_key in features:
                features[f'市场变化_未来1h减当前_{name}'] = features[future_key] - features[current_key]

    def _add_market_daily_shape_features(self, features: Dict, date_data: pd.DataFrame, target_hour: int) -> None:
        """添加目标日市场边界日内形态统计，不使用价格列。"""
        hourly_values = []
        for hour in range(24):
            hour_data = date_data[date_data['小时'] == hour]
            if len(hour_data) == 1:
                hourly_values.append(self._compute_market_values(hour_data.iloc[0]))

        if len(hourly_values) != 24:
            return

        for name in ['系统负荷', '新能源出力', '净负荷', '剩余负荷', '市场化缺口', '供需覆盖率']:
            values = [item[name] for item in hourly_values if name in item and pd.notna(item[name])]
            if len(values) != 24:
                continue

            arr = np.asarray(values, dtype=float)
            current = hourly_values[target_hour].get(name)
            features[f'市场日形态_{name}_均值'] = float(arr.mean())
            features[f'市场日形态_{name}_标准差'] = float(arr.std())
            features[f'市场日形态_{name}_最小值'] = float(arr.min())
            features[f'市场日形态_{name}_最大值'] = float(arr.max())
            if current is not None:
                features[f'市场日形态_H{target_hour:02d}_{name}_相对日均值'] = float(current - arr.mean())
                denominator = arr.max() - arr.min()
                features[f'市场日形态_H{target_hour:02d}_{name}_日内位置'] = (
                    float((current - arr.min()) / denominator) if denominator != 0 else 0.0
                )

    def _add_midday_market_shape_features(self, features: Dict, date_data: pd.DataFrame, target_hour: int) -> None:
        """添加目标日H08-H15市场形态，聚焦午间低价风险。"""
        if target_hour < 8 or target_hour > 15:
            return

        hourly_values = []
        for hour in range(8, 16):
            hour_data = date_data[date_data['小时'] == hour]
            if len(hour_data) != 1:
                return
            hourly_values.append((hour, self._compute_market_values(hour_data.iloc[0])))

        for name in ['系统负荷', '光伏出力', '新能源出力', '净负荷', '剩余负荷', '市场化缺口', '供需覆盖率']:
            values = []
            current = None
            for hour, item in hourly_values:
                if name not in item or pd.isna(item[name]):
                    continue
                value = float(item[name])
                values.append(value)
                if hour == target_hour:
                    current = value
            if len(values) != 8 or current is None:
                continue

            arr = np.asarray(values, dtype=float)
            prefix = f'午间市场形态_{name}'
            features[f'{prefix}_均值'] = float(arr.mean())
            features[f'{prefix}_标准差'] = float(arr.std())
            features[f'{prefix}_最小值'] = float(arr.min())
            features[f'{prefix}_最大值'] = float(arr.max())
            features[f'{prefix}_H08到H15爬坡'] = float(arr[-1] - arr[0])
            denominator = arr.max() - arr.min()
            features[f'午间市场形态_H{target_hour:02d}_{name}_相对午间均值'] = float(current - arr.mean())
            features[f'午间市场形态_H{target_hour:02d}_{name}_午间位置'] = (
                float((current - arr.min()) / denominator) if denominator != 0 else 0.0
            )

    def _add_midday_weather_agg_features(self, features: Dict, date_data: pd.DataFrame, target_hour: int) -> None:
        """添加聚合气象特征，减少直接使用数百个气象源的过拟合风险。"""
        weather_keywords = ['温度', '总云量', '辐照度']
        weather_cols = [
            col for col in date_data.columns
            if any(keyword in col for keyword in weather_keywords)
        ]
        if not weather_cols:
            return

        for prefix, hour in [
            ('当前', target_hour),
            ('滞后1h', target_hour - 1 if target_hour > 0 else 23),
            ('未来1h', target_hour + 1 if target_hour < 23 else 0),
        ]:
            hour_data = date_data[date_data['小时'] == hour]
            if len(hour_data) != 1:
                continue
            row = hour_data.iloc[0]
            self._add_weather_aggregates(features, row, prefix, weather_cols)

        if 8 <= target_hour <= 15:
            midday_data = date_data[date_data['小时'].between(8, 15)]
            if len(midday_data) == 8:
                for keyword in weather_keywords:
                    cols = [col for col in weather_cols if keyword in col]
                    values = midday_data[cols].to_numpy(dtype=float).reshape(-1) if cols else np.asarray([])
                    values = values[~np.isnan(values)]
                    if values.size == 0:
                        continue
                    features[f'午间气象聚合_{keyword}_均值'] = float(values.mean())
                    features[f'午间气象聚合_{keyword}_最大值'] = float(values.max())
                    features[f'午间气象聚合_{keyword}_标准差'] = float(values.std())

    def _add_weather_aggregates(self, features: Dict, row: pd.Series, prefix: str, weather_cols: List[str]) -> None:
        for keyword in ['温度', '总云量', '辐照度']:
            values = []
            for col in weather_cols:
                if keyword in col and col in row.index and pd.notna(row[col]):
                    values.append(float(row[col]))
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            features[f'{prefix}_气象聚合_{keyword}_均值'] = float(arr.mean())
            features[f'{prefix}_气象聚合_{keyword}_最大值'] = float(arr.max())
            features[f'{prefix}_气象聚合_{keyword}_最小值'] = float(arr.min())
            features[f'{prefix}_气象聚合_{keyword}_标准差'] = float(arr.std())

    @staticmethod
    def _sum_available(*values: Optional[float]) -> Optional[float]:
        valid = [float(value) for value in values if value is not None and pd.notna(value)]
        if not valid:
            return None
        return float(np.sum(valid))

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator is None or pd.isna(denominator) or abs(denominator) < 1e-6:
            return 0.0
        return float(numerator / denominator)
    
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
            'description': '滑动窗口Direct多步预测，实时价格仅T-2及之前，市场/气象特征使用当前、1h滞后和1h未来'
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
