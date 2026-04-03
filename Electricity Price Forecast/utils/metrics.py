"""
评估指标模块
包含电价预测任务中使用的各种评估指标
"""

import numpy as np
from typing import Union


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 (Mean Squared Error)
    
    MSE = (1/n) * sum((y_true - y_pred)^2)
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    mse : float
        均方误差
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Squared Error)
    
    RMSE = sqrt(MSE)
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    rmse : float
        均方根误差
    """
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    MAE = (1/n) * sum(|y_true - y_pred|)
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    mae : float
        平均绝对误差
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    MAPE = (100%/n) * sum(|(y_true - y_pred) / y_true|)
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    mape : float
        平均绝对百分比误差 (%)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 避免除以零
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error)
    
    sMAPE = (100%/n) * sum(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    优点: 对称处理，避免分母为零问题
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    smape : float
        对称平均绝对百分比误差 (%)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # 避免除以零
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R-squared)
    
    R² = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    r2 : float
        决定系数
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算所有评估指标
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
        
    Returns
    -------
    metrics : dict
        包含所有指标的字典
    """
    return {
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'sMAPE': calculate_smape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred)
    }


def calculate_hourly_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_func) -> np.ndarray:
    """
    计算每小时的评估指标
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, 24)
        真实值，每行代表一天的24小时
    y_pred : np.ndarray, shape (n_samples, 24)
        预测值，每行代表一天的24小时
    metric_func : callable
        评估指标函数
        
    Returns
    -------
    hourly_metrics : np.ndarray
        每小时的指标值
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_hours = y_true.shape[1] if len(y_true.shape) > 1 else 1
    hourly_metrics = np.zeros(n_hours)
    
    for h in range(n_hours):
        if len(y_true.shape) > 1:
            hourly_metrics[h] = metric_func(y_true[:, h], y_pred[:, h])
        else:
            hourly_metrics[h] = metric_func(y_true, y_pred)
    
    return hourly_metrics
