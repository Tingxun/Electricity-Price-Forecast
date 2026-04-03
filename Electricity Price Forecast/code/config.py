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
        
        # 数据路径配置
        self.data_paths = {
            'raw_data': self.project_root / '湖北省能源数据集',
            'processed': self.project_root / 'data' / 'processed',
            'features': self.project_root / 'data' / 'features',
            'raw': self.project_root / 'data' / 'raw'
        }
        
        # 模型保存路径
        self.model_paths = {
            'linear': self.project_root / 'saved_models' / 'linear',
            'tree': self.project_root / 'saved_models' / 'tree',
            'neural': self.project_root / 'saved_models' / 'neural',
            'epftoolbox': self.project_root / 'saved_models' / 'epftoolbox'
        }
        
        # 结果输出路径
        self.result_paths = {
            'predictions': self.project_root / 'results' / 'predictions',
            'figures': self.project_root / 'results' / 'figures',
            'logs': self.project_root / 'results' / 'logs'
        }
        
        # 数据时间范围
        self.data_config = {
            'start_date': '2024-04-16',
            'end_date': '2025-03-28',
            'test_start_date': '2025-03-01',  # 测试集开始日期（最后一个月）
            'hours_per_day': 24,
            'forecast_horizon': 24  # 预测未来24小时
        }
        
        # 数据集划分配置
        self.split_config = {
            'test_size_days': 28,  # 测试集28天（约1个月）
            'validation_size_days': 14,  # 验证集14天
            'random_seed': 42
        }
        
        # 特征工程配置
        self.feature_config = {
            'lag_periods': [1, 2, 3, 7],  # 滞后特征阶数
            'rolling_windows': [7, 14, 30],  # 滚动窗口大小
            'rolling_aggregations': ['mean', 'std', 'min', 'max'],
            'normalize_method': 'standard',  # 标准化方法
            'handle_missing_method': 'interpolate'  # 缺失值处理方法
        }
        
        # 气象数据聚类配置
        self.weather_cluster_config = {
            'enabled': True,
            'n_clusters': 5,  # 聚类数量
            'cluster_method': 'kmeans',  # kmeans, gmm, hierarchical
            'features': ['temperature_mean', 'temperature_max', 'temperature_min', 
                        'humidity_mean', 'wind_speed_mean']
        }
        
        # 特征优化配置
        self.optimization_config = {
            'enabled': False,  # 暂时不启用特征优化，只进行超参数优化
            'method': 'pso',  # pso, ga, bayesian
            'pso_params': {
                'n_particles': 30,
                'n_iterations': 100,
                'w': 0.9,  # 惯性权重
                'c1': 2.0,  # 个体学习因子
                'c2': 2.0   # 群体学习因子
            },
            'ga_params': {
                'population_size': 50,
                'n_generations': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1
            },
            'bayesian_params': {
                'n_iter': 50,
                'cv': 5
            }
        }
        
        # 模型配置
        self.model_config = {
            'enabled_models': ['linear', 'tree'],  # 启用的模型类型
            'linear_models': {
                'LinearRegression': {
                    'enabled': True,
                    'params': {}
                },
                'Ridge': {
                    'enabled': True,
                    'params': {
                        'alpha': 1.0,
                        'random_state': 42
                    }
                },
                'Lasso': {
                    'enabled': True,
                    'params': {
                        'alpha': 0.1,
                        'random_state': 42
                    }
                }
            },
            'tree_models': {
                'XGBoost': {
                    'enabled': True,
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': 42
                    }
                },
                'LightGBM': {
                    'enabled': True,
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': 42
                    }
                },
                'CatBoost': {
                    'enabled': False,  # 可选
                    'params': {
                        'iterations': 100,
                        'depth': 6,
                        'learning_rate': 0.1,
                        'random_seed': 42,
                        'verbose': False
                    }
                }
            },
            'neural_models': {
                'enabled': False,  # 深度学习模型默认不启用
                'MLP': {
                    'hidden_layers': [128, 64, 32],
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            },
            'epftoolbox_models': {
                'enabled': False,  # epftoolbox作为可选依赖
                'LEAR': {
                    'calibration_window': 728,  # 2年
                    'lasso_alpha': 0.1
                },
                'DNN': {
                    'n_layers': 2,
                    'neurons_per_layer': 50,
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            }
        }
        
        # 评估指标配置
        self.metrics_config = {
            'primary_metrics': ['MSE', 'MAE', 'sMAPE'],
            'secondary_metrics': ['RMSE', 'MAPE', 'R2'],
            'calculate_per_hour': True  # 是否计算每小时的指标
        }
        
        # 训练配置
        self.training_config = {
            'mode': 'train',  # train, predict, evaluate
            'cv_folds': 5,
            'early_stopping_rounds': 50,
            'verbose': True
        }
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        for path_dict in [self.data_paths, self.model_paths, self.result_paths]:
            for path in path_dict.values():
                path.mkdir(parents=True, exist_ok=True)
    
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
        for model_type in ['linear_models', 'tree_models', 'neural_models', 'epftoolbox_models']:
            if model_name in self.model_config.get(model_type, {}):
                return self.model_config[model_type][model_name].get('params', {})
        return {}
    
    def is_model_enabled(self, model_name: str) -> bool:
        """检查模型是否启用"""
        for model_type in ['linear_models', 'tree_models', 'neural_models', 'epftoolbox_models']:
            if model_name in self.model_config.get(model_type, {}):
                return self.model_config[model_type][model_name].get('enabled', False)
        return False
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"未知的配置项: {key}")


# 全局配置实例
config = Config()
