"""
数据处理工具模块
包含数据清洗、缺失值处理、异常值检测等功能
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    加载数据文件
    
    Parameters
    ----------
    file_path : str
        数据文件路径
    **kwargs : dict
        传递给pd.read_csv或pd.read_excel的参数
        
    Returns
    -------
    data : pd.DataFrame
        加载的数据
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def handle_missing_values(
    df: pd.DataFrame, 
    method: str = 'interpolate',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    处理缺失值
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    method : str
        处理方法: 'interpolate'(插值), 'forward'(前向填充), 
                'backward'(后向填充), 'mean'(均值填充), 'drop'(删除)
    limit : int, optional
        连续缺失值的最大填充数量
        
    Returns
    -------
    df : pd.DataFrame
        处理后的数据
    """
    df = df.copy()
    
    if method == 'interpolate':
        df = df.interpolate(method='linear', limit=limit)
        # 边界值使用前后填充
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'forward':
        df = df.fillna(method='ffill', limit=limit)
    elif method == 'backward':
        df = df.fillna(method='bfill', limit=limit)
    elif method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"未知的缺失值处理方法: {method}")
    
    return df


def detect_outliers(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    检测异常值
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    columns : list, optional
        要检测的列，None表示所有数值列
    method : str
        检测方法: 'iqr'(四分位距法), 'zscore'(Z分数法)
    threshold : float
        异常值判定阈值
        
    Returns
    -------
    outlier_mask : pd.DataFrame
        异常值标记，True表示异常值
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outlier_mask[col] = z_scores > threshold
    
    return outlier_mask


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'clip',
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.DataFrame:
    """
    处理异常值
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    columns : list, optional
        要处理的列
    method : str
        处理方法: 'clip'(截断), 'remove'(删除), 'nan'(置为缺失)
    lower_percentile : float
        下分位数
    upper_percentile : float
        上分位数
        
    Returns
    -------
    df : pd.DataFrame
        处理后的数据
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)
        
        if method == 'clip':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]
        elif method == 'nan':
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
    
    return df


def normalize_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, dict]:
    """
    特征标准化/归一化
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    columns : list, optional
        要标准化的列
    method : str
        标准化方法: 'standard'(Z-score), 'minmax'(Min-Max), 'robust'(稳健标准化)
        
    Returns
    -------
    df_normalized : pd.DataFrame
        标准化后的数据
    scaler_params : dict
        标准化参数，用于后续反变换
    """
    df = df.copy()
    scaler_params = {}
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        
        if method == 'standard':
            mean = data.mean()
            std = data.std()
            if std > 0:
                df[col] = (df[col] - mean) / std
                scaler_params[col] = {'method': 'standard', 'mean': mean, 'std': std}
                
        elif method == 'minmax':
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                scaler_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
        elif method == 'robust':
            median = data.median()
            iqr = data.quantile(0.75) - data.quantile(0.25)
            if iqr > 0:
                df[col] = (df[col] - median) / iqr
                scaler_params[col] = {'method': 'robust', 'median': median, 'iqr': iqr}
    
    return df, scaler_params


def inverse_normalize(
    df: pd.DataFrame,
    scaler_params: dict
) -> pd.DataFrame:
    """
    反标准化
    
    Parameters
    ----------
    df : pd.DataFrame
        标准化后的数据
    scaler_params : dict
        标准化参数
        
    Returns
    -------
    df_original : pd.DataFrame
        反标准化后的数据
    """
    df = df.copy()
    
    for col, params in scaler_params.items():
        if col not in df.columns:
            continue
            
        if params['method'] == 'standard':
            df[col] = df[col] * params['std'] + params['mean']
        elif params['method'] == 'minmax':
            df[col] = df[col] * (params['max'] - params['min']) + params['min']
        elif params['method'] == 'robust':
            df[col] = df[col] * params['iqr'] + params['median']
    
    return df


def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    创建时间特征
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    datetime_col : str
        时间列名
        
    Returns
    -------
    df : pd.DataFrame
        包含时间特征的数据
    """
    df = df.copy()
    
    # 确保时间列为datetime类型
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # 提取时间特征
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['month'] = df[datetime_col].dt.month
    df['day_of_year'] = df[datetime_col].dt.dayofyear
    df['week_of_year'] = df[datetime_col].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
    
    # 季节特征 (1:春, 2:夏, 3:秋, 4:冬)
    df['season'] = df['month'].map({3: 1, 4: 1, 5: 1,
                                     6: 2, 7: 2, 8: 2,
                                     9: 3, 10: 3, 11: 3,
                                     12: 4, 1: 4, 2: 4})
    
    # 正弦/余弦编码（保持时间周期性）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: List[int],
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    创建滞后特征
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    column : str
        要创建滞后特征的列
    lags : list
        滞后阶数列表
    group_col : str, optional
        分组列名（用于分组滞后）
        
    Returns
    -------
    df : pd.DataFrame
        包含滞后特征的数据
    """
    df = df.copy()
    
    if group_col is not None:
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df.groupby(group_col)[column].shift(lag)
    else:
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int],
    aggregations: List[str] = ['mean', 'std', 'min', 'max'],
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    创建滚动统计特征
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    column : str
        要创建滚动特征的列
    windows : list
        窗口大小列表
    aggregations : list
        聚合方法列表
    group_col : str, optional
        分组列名
        
    Returns
    -------
    df : pd.DataFrame
        包含滚动特征的数据
    """
    df = df.copy()
    
    for window in windows:
        for agg in aggregations:
            col_name = f'{column}_rolling_{agg}_{window}'
            
            if group_col is not None:
                df[col_name] = df.groupby(group_col)[column].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).agg(agg)
                )
            else:
                df[col_name] = df[column].shift(1).rolling(window=window, min_periods=1).agg(agg)
    
    return df
