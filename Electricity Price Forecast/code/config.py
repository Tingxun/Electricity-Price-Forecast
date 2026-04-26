"""
配置文件
包含项目路径、模型参数、数据配置等
"""

import os
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """
    项目配置类
    """
    
    def __init__(self):
        # 项目根目录
        self.project_root = Path(__file__).parent.parent
        
        # 数据路径配置 - 使用预处理完成的完整数据集
        self.data_paths = {
            'processed_data': self.project_root / 'data' / 'processed' / 'processed_data.csv',
            'features': self.project_root / 'data' / 'features',
            'feature_info': self.project_root / 'data' / 'features' / 'feature_info.json'
        }
        
        # 模型保存路径
        self.model_paths = {
            'linear': self.project_root / 'saved_models' / 'linear',
            'tree': self.project_root / 'saved_models' / 'tree',
            'neural': self.project_root / 'saved_models' / 'neural'
        }
        
        # 结果输出路径
        self.result_paths = {
            'predictions': self.project_root / 'results' / 'predictions',
            'figures': self.project_root / 'results' / 'figures',
            'logs': self.project_root / 'results' / 'logs'
        }
        
        # 数据时间范围
        self.data_config = {
            'start_date': '2024-05-28',
            'end_date': '2025-03-26',
            'test_start_date': '2025-03-01',
            'hours_per_day': 24,
            'forecast_horizon': 24
        }
        
        # 数据集划分配置
        self.split_config = {
            'test_size_days': 26,
            'validation_size_days': 14,
            'random_seed': 42
        }
        
        # 特征工程配置
        self.feature_config = {
            'lag_periods': [1, 2, 3, 7],
            'rolling_windows': [7, 14, 30],
            'rolling_aggregations': ['mean', 'std', 'min', 'max'],
            'normalize_method': 'standard',
            'handle_missing_method': 'interpolate'
        }
        
        # 模型配置
        self.model_config = {
            'enabled_models': ['linear', 'tree'],
            'linear_models': {
                'LinearRegression': {'enabled': True, 'params': {}},
                'Ridge': {'enabled': True, 'params': {'alpha': 1.0, 'random_state': 42}},
                'Lasso': {'enabled': True, 'params': {'alpha': 0.1, 'random_state': 42}}
            },
            'tree_models': {
                'XGBoost': {'enabled': True, 'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}},
                'LightGBM': {'enabled': True, 'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}},
                'RandomForest': {'enabled': True, 'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}}
            },
            'neural_models': {
                'enabled': False,
                'MLP': {'hidden_layers': [128, 64, 32], 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100}
            }
        }
        
        # 评估指标配置
        self.metrics_config = {
            'primary_metrics': ['MSE', 'MAE', 'sMAPE'],
            'secondary_metrics': ['RMSE', 'R2'],
            'calculate_per_hour': True
        }
        
        # 训练配置
        self.training_config = {
            'mode': 'train',
            'cv_folds': 5,
            'early_stopping_rounds': 50,
            'verbose': True
        }
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        for path_dict in [self.model_paths, self.result_paths]:
            for path in path_dict.values():
                path.mkdir(parents=True, exist_ok=True)
        self.data_paths['features'].mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, key: str) -> Path:
        """获取数据路径"""
        return self.data_paths.get(key)
    
    def get_model_path(self, model_type: str) -> Path:
        """获取模型保存路径"""
        return self.model_paths.get(model_type)
    
    def get_result_path(self, result_type: str) -> Path:
        """获取结果输出路径"""
        return self.result_paths.get(result_type)
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """获取模型参数"""
        for model_type in ['linear_models', 'tree_models', 'neural_models']:
            if model_name in self.model_config.get(model_type, {}):
                return self.model_config[model_type][model_name].get('params', {})
        return {}
    
    def is_model_enabled(self, model_name: str) -> bool:
        """检查模型是否启用"""
        for model_type in ['linear_models', 'tree_models', 'neural_models']:
            if model_name in self.model_config.get(model_type, {}):
                return self.model_config[model_type][model_name].get('enabled', False)
        return False


# 全局配置实例
config = Config()
