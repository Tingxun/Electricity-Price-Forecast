"""
数据加载模块
负责加载和初步处理市场边界信息和出清价格数据

根据实验方案要求，支持：
- 历史实时数据（T-2及之前）
- 日前披露数据（T日披露T+1）
- 天气聚类标签集成
- 数据质量检查和验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys

sys.path.append(str(Path(__file__).parent))
from config import config


class DataLoader:
    """
    数据加载器
    加载市场边界信息和出清价格数据
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化数据加载器
        
        Parameters
        ----------
        data_path : str, optional
            数据文件路径，默认使用config中的路径
        """
        if data_path is None:
            self.data_path = config.data_paths['raw'] / '市场边界_出清价格总表.csv'
        else:
            self.data_path = Path(data_path)
        
        self.raw_data = None
        self.processed_data = None
        
        # 定义数据列分类（根据实验方案）
        self._define_column_categories()
    
    def _define_column_categories(self):
        """定义数据列分类"""
        # 实时数据列（T-2及之前）
        self.real_time_cols = [
            '系统负荷-实时', '风电出力-实时', '光伏出力-实时', 
            '水电出力-实时', '联络线计划-实时', '非市场化机组出力-实时',
            '平均出清价格-实时（元/MWh）'
        ]
        
        # 日前数据列（T日披露T+1）
        self.day_ahead_cols = [
            '系统负荷-日前', '风电出力-日前', '光伏出力-日前',
            '水电出力-日前', '联络线计划-日前', '非市场化机组出力-日前',
            '平均出清价格-日前（元/MWh）'
        ]
        
        # 时间特征列
        self.time_cols = [
            '年', '月', '日', '星期', '是否周末', '季度', '小时', 
            '是否高峰时段', '是否夜间'
        ]
        
        # 所有特征列（不包括目标变量）
        self.feature_cols = self.time_cols + [
            col for col in self.real_time_cols + self.day_ahead_cols 
            if col != '平均出清价格-实时（元/MWh）'
        ]
        
        # 目标变量
        self.target_cols = ['平均出清价格-实时（元/MWh）']
        
    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns
        -------
        df : pd.DataFrame
            加载的数据
        """
        print(f"正在加载数据: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # 转换日期列
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 创建完整的时间戳
        df['datetime'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['时段'].str.split('-').str[0])
        
        # 按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        self.raw_data = df.copy()
        
        print(f"数据加载完成，共 {len(df)} 条记录")
        print(f"时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
        
        return df
    
    def get_feature_columns(self, data_type: str = 'all') -> list:
        """
        获取特征列名
        
        Parameters
        ----------
        data_type : str
            数据类型：'all', 'real_time', 'day_ahead', 'time'
            
        Returns
        -------
        feature_cols : list
            特征列名列表
        """
        if data_type == 'real_time':
            return [col for col in self.real_time_cols if col != '平均出清价格-实时（元/MWh）']
        elif data_type == 'day_ahead':
            return [col for col in self.day_ahead_cols if col != '平均出清价格-日前（元/MWh）']
        elif data_type == 'time':
            return self.time_cols
        else:
            return self.feature_cols
    
    def get_target_columns(self) -> list:
        """
        获取目标变量列名
        
        Returns
        -------
        target_cols : list
            目标变量列名列表
        """
        return self.target_cols
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        检查数据质量
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        quality_report : dict
            数据质量报告
        """
        report = {
            'total_rows': len(df),
            'date_range': (df['日期'].min(), df['日期'].max()),
            'missing_values': {},
            'missing_percentage': {},
            'zero_values': {},
            'negative_values': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 缺失值统计
            missing_count = df[col].isna().sum()
            report['missing_values'][col] = missing_count
            report['missing_percentage'][col] = missing_count / len(df) * 100
            
            # 零值统计
            zero_count = (df[col] == 0).sum()
            report['zero_values'][col] = zero_count
            
            # 负值统计
            negative_count = (df[col] < 0).sum()
            report['negative_values'][col] = negative_count
        
        return report
    
    def split_train_test(self, df: pd.DataFrame, test_start_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间划分训练集和测试集
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        test_start_date : str, optional
            测试集开始日期，默认使用config中的配置
            
        Returns
        -------
        train_df, test_df : tuple
            训练集和测试集
        """
        if test_start_date is None:
            test_start_date = config.data_config['test_start_date']
        
        test_start = pd.to_datetime(test_start_date)
        
        train_df = df[df['日期'] < test_start].copy()
        test_df = df[df['日期'] >= test_start].copy()
        
        print(f"训练集: {train_df['日期'].min()} 至 {train_df['日期'].max()}, 共 {len(train_df)} 条")
        print(f"测试集: {test_df['日期'].min()} 至 {test_df['日期'].max()}, 共 {len(test_df)} 条")
        
        return train_df, test_df
    
    def get_day_ahead_features(self, df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        """
        获取指定日期的日前市场边界信息（用于预测T+1）
        
        Parameters
        ----------
        df : pd.DataFrame
            完整数据集
        target_date : pd.Timestamp
            目标日期（T+1）
            
        Returns
        -------
        day_ahead_features : pd.DataFrame
            该日期的日前市场边界信息（24小时）
        """
        # 筛选目标日期的数据
        day_data = df[df['日期'] == target_date].copy()
        
        if len(day_data) == 0:
            raise ValueError(f"未找到日期 {target_date} 的数据")
        
        # 日前特征列（以"-日前"结尾的列）
        day_ahead_cols = [col for col in df.columns if '-日前' in col]
        
        # 时间特征列
        time_cols = ['时段', '小时', '是否高峰时段', '是否夜间', '星期', '是否周末', '季度']
        
        feature_cols = time_cols + day_ahead_cols
        
        return day_data[feature_cols].reset_index(drop=True)
    
    def get_historical_real_time_data(self, df: pd.DataFrame, T_date: pd.Timestamp, 
                                    lookback_days: int = 60) -> pd.DataFrame:
        """
        获取历史实时数据（T-2及之前）
        
        Parameters
        ----------
        df : pd.DataFrame
            完整数据集
        T_date : pd.Timestamp
            T日期
        lookback_days : int
            回溯天数，默认60天
            
        Returns
        -------
        historical_data : pd.DataFrame
            历史实时数据（T-2及之前）
        """
        T_minus_2 = T_date - pd.Timedelta(days=2)
        start_date = T_minus_2 - pd.Timedelta(days=lookback_days)
        
        # 获取T-2及之前的数据
        historical_data = df[
            (df['日期'] >= start_date) & (df['日期'] <= T_minus_2)
        ].copy()
        
        # 只保留实时数据列、时间列和日期列
        real_time_cols = self.get_feature_columns('real_time') + self.time_cols + self.target_cols + ['日期']
        historical_data = historical_data[real_time_cols]
        
        print(f"历史实时数据: {start_date.date()} 至 {T_minus_2.date()}, 共 {len(historical_data)} 条记录")
        
        return historical_data.reset_index(drop=True)
    
    def get_day_ahead_disclosure_data(self, df: pd.DataFrame, T_date: pd.Timestamp, 
                                    target_date: pd.Timestamp) -> pd.DataFrame:
        """
        获取日前披露数据（T日披露T+1信息）
        
        Parameters
        ----------
        df : pd.DataFrame
            完整数据集
        T_date : pd.Timestamp
            T日期（披露日期）
        target_date : pd.Timestamp
            目标日期（T+1）
            
        Returns
        -------
        day_ahead_data : pd.DataFrame
            日前披露数据
        """
        # 获取T+1日的日前数据
        day_ahead_data = df[df['日期'] == target_date].copy()
        
        if len(day_ahead_data) == 0:
            raise ValueError(f"未找到日期 {target_date} 的日前数据")
        
        # 只保留日前数据列和时间列
        day_ahead_cols = self.get_feature_columns('day_ahead') + self.time_cols
        day_ahead_data = day_ahead_data[day_ahead_cols]
        
        print(f"日前披露数据: {target_date.date()} (T+1), 共 {len(day_ahead_data)} 条记录")
        
        return day_ahead_data.reset_index(drop=True)
    
    def add_weather_cluster_labels(self, df: pd.DataFrame, weather_cluster_path: Optional[str] = None) -> pd.DataFrame:
        """
        添加天气聚类标签（根据实验方案）
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        weather_cluster_path : str, optional
            天气聚类标签文件路径
            
        Returns
        -------
        df_with_cluster : pd.DataFrame
            包含天气聚类标签的数据
        """
        df_with_cluster = df.copy()
        
        # 如果提供了天气聚类文件，加载并合并
        if weather_cluster_path and Path(weather_cluster_path).exists():
            try:
                weather_cluster = pd.read_csv(weather_cluster_path)
                weather_cluster['日期'] = pd.to_datetime(weather_cluster['日期'])
                
                # 合并天气聚类标签
                df_with_cluster = pd.merge(
                    df_with_cluster, 
                    weather_cluster[['日期', '天气聚类标签']], 
                    on='日期', 
                    how='left'
                )
                print("天气聚类标签已添加")
            except Exception as e:
                print(f"天气聚类标签加载失败: {e}")
        else:
            # 如果没有天气聚类文件，创建占位符列
            df_with_cluster['天气聚类标签'] = -1
            print("未找到天气聚类文件，使用默认标签")
        
        return df_with_cluster
    
    def prepare_prediction_data(self, df: pd.DataFrame, prediction_date: pd.Timestamp, 
                              lookback_days: int = 60, weather_cluster_path: Optional[str] = None) -> dict:
        """
        准备预测所需的数据
        
        在T时间点，准备T+1预测所需的数据：
        - 历史实时数据（T-2及之前）
        - 日前披露数据（T日披露T+1信息）
        - 天气聚类标签（可选）
        
        Parameters
        ----------
        df : pd.DataFrame
            完整数据集
        prediction_date : pd.Timestamp
            预测目标日期（T+1）
        lookback_days : int
            历史数据回溯天数
        weather_cluster_path : str, optional
            天气聚类标签文件路径
            
        Returns
        -------
        data_dict : dict
            包含历史数据和日前数据的字典
        """
        T_date = prediction_date - pd.Timedelta(days=1)  # T日期
        
        # 获取历史实时数据（T-2及之前）
        historical_data = self.get_historical_real_time_data(df, T_date, lookback_days)
        
        # 获取日前披露数据（T日披露的T+1信息）
        try:
            day_ahead_data = self.get_day_ahead_disclosure_data(df, T_date, prediction_date)
        except ValueError:
            print(f"警告: 未找到 {prediction_date} 的日前数据")
            day_ahead_data = None
        
        # 添加天气聚类标签
        if weather_cluster_path:
            historical_data = self.add_weather_cluster_labels(historical_data, weather_cluster_path)
            if day_ahead_data is not None:
                day_ahead_data = self.add_weather_cluster_labels(day_ahead_data, weather_cluster_path)
        
        # 数据质量检查
        historical_quality = self.check_prediction_data_quality(historical_data, 'historical')
        day_ahead_quality = self.check_prediction_data_quality(day_ahead_data, 'day_ahead') if day_ahead_data is not None else {}
        
        return {
            'historical_data': historical_data,
            'day_ahead_data': day_ahead_data,
            'prediction_date': prediction_date,
            'T_date': T_date,
            'T_minus_2': T_date - pd.Timedelta(days=2),
            'data_quality': {
                'historical': historical_quality,
                'day_ahead': day_ahead_quality
            }
        }
    
    def check_prediction_data_quality(self, df: pd.DataFrame, data_type: str) -> dict:
        """
        检查预测数据的质量
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
        data_type : str
            数据类型：'historical' 或 'day_ahead'
            
        Returns
        -------
        quality_report : dict
            数据质量报告
        """
        if df is None or len(df) == 0:
            return {'status': 'empty', 'message': '数据为空'}
        
        report = {
            'status': 'ok',
            'data_type': data_type,
            'total_rows': len(df),
            'missing_columns': [],
            'missing_values': {},
            'date_range': (df['日期'].min(), df['日期'].max()) if '日期' in df.columns else (None, None)
        }
        
        # 检查必需列是否存在
        required_cols = []
        if data_type == 'historical':
            required_cols = self.get_feature_columns('real_time') + self.time_cols + self.target_cols
        elif data_type == 'day_ahead':
            required_cols = self.get_feature_columns('day_ahead') + self.time_cols
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report['status'] = 'warning'
            report['missing_columns'] = missing_cols
        
        # 检查缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report['missing_values'][col] = missing_count
                report['status'] = 'warning'
        
        # 检查数据完整性（24小时数据）
        if '小时' in df.columns:
            unique_hours = df['小时'].nunique()
            if unique_hours != 24:
                report['status'] = 'warning'
                report['incomplete_hours'] = f"只有 {unique_hours}/24 小时数据"
        
        return report
    
    def validate_data_for_training(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        验证数据是否适合训练
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        is_valid : bool
            数据是否有效
        message : str
            验证结果消息
        """
        if df is None or len(df) == 0:
            return False, "数据为空"
        
        # 检查目标变量
        target_cols = self.get_target_columns()
        for target_col in target_cols:
            if target_col not in df.columns:
                return False, f"缺少目标变量: {target_col}"
            
            # 检查目标变量是否有足够非空值
            non_null_count = df[target_col].notna().sum()
            if non_null_count < len(df) * 0.8:  # 至少80%非空
                return False, f"目标变量 {target_col} 缺失值过多"
        
        # 检查时间连续性
        if '日期' in df.columns and '小时' in df.columns:
            # 检查是否有重复的时间点
            time_duplicates = df.duplicated(subset=['日期', '小时']).sum()
            if time_duplicates > 0:
                return False, f"发现 {time_duplicates} 个重复时间点"
        
        return True, "数据验证通过"


def main():
    """
    测试数据加载器功能
    """
    print("=== 电价预测数据加载器测试 ===")
    
    # 初始化数据加载器
    loader = DataLoader()
    
    # 加载数据
    df = loader.load_data()
    
    # 数据质量检查
    quality_report = loader.check_data_quality(df)
    print("\n数据质量报告:")
    print(f"总记录数: {quality_report['total_rows']}")
    print(f"时间范围: {quality_report['date_range'][0]} 至 {quality_report['date_range'][1]}")
    
    # 检查缺失值
    print("\n缺失值统计:")
    for col, count in quality_report['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count} 个缺失值 ({quality_report['missing_percentage'][col]:.2f}%)")
    
    # 测试数据划分
    train_df, test_df = loader.split_train_test(df)
    
    # 测试预测数据准备
    test_date = pd.to_datetime('2025-03-01')  # 测试集开始日期
    prediction_data = loader.prepare_prediction_data(df, test_date)
    
    print(f"\n预测数据准备完成:")
    print(f"预测日期: {prediction_data['prediction_date'].date()}")
    print(f"T日期: {prediction_data['T_date'].date()}")
    print(f"T-2日期: {prediction_data['T_minus_2'].date()}")
    print(f"历史数据记录数: {len(prediction_data['historical_data'])}")
    print(f"日前数据记录数: {len(prediction_data['day_ahead_data']) if prediction_data['day_ahead_data'] is not None else 0}")
    
    # 验证训练数据
    is_valid, message = loader.validate_data_for_training(train_df)
    print(f"\n训练数据验证: {is_valid} - {message}")
    
    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    main()
