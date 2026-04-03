"""
超参数优化脚本
用于优化模型超参数
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import create_linear_model, create_tree_model, create_neural_network
from utils.metrics import calculate_mae

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    超参数调优器
    支持网格搜索、随机搜索和贝叶斯优化
    """
    
    def __init__(self, config: Config):
        """
        初始化调优器
        
        Parameters
        ----------
        config : Config
            项目配置对象
        """
        self.config = config
        
        # 创建保存目录
        os.makedirs(config.result_paths['logs'], exist_ok=True)
        
        logger.info("超参数调优器初始化完成")
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        准备数据
        
        Returns
        -------
        data_dict : dict
            数据字典
        """
        logger.info("开始准备数据...")
        
        # 加载数据
        loader = DataLoader(self.config)
        df = loader.load_data()
        
        # 特征工程
        engineer = FeatureEngineer()
        features_df, target_cols = engineer.create_all_features(df)
        
        # 准备特征和目标
        feature_cols = [col for col in features_df.columns if col not in target_cols + ['Date', 'Hour']]
        X = features_df[feature_cols].values
        y = features_df[target_cols].values
        
        # 时间序列划分
        dates = features_df['Date'].unique()
        n_dates = len(dates)
        
        train_end = int(n_dates * 0.8)
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:]
        
        train_mask = features_df['Date'].isin(train_dates)
        val_mask = features_df['Date'].isin(val_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        logger.info(f"数据准备完成:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_val)} 样本")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'feature_cols': feature_cols
        }
    
    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型的参数搜索空间
        
        Parameters
        ----------
        model_name : str
            模型名称
            
        Returns
        -------
        param_grid : dict
            参数搜索空间
        """
        param_grids = {
            'Ridge': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'ElasticNet': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'DecisionTree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def grid_search(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        网格搜索
        
        Parameters
        ----------
        model_name : str
            模型名称
        data : dict
            数据字典
            
        Returns
        -------
        result : dict
            搜索结果
        """
        logger.info(f"开始网格搜索: {model_name}")
        
        param_grid = self.get_param_grid(model_name)
        if not param_grid:
            logger.warning(f"模型 {model_name} 没有预定义的参数网格")
            return {}
        
        # 创建基础模型
        if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            model_type = model_name.lower()
            if model_name == 'ElasticNet':
                model_type = 'elastic_net'
            base_model = create_linear_model(model_type=model_type)
        elif model_name in ['DecisionTree', 'RandomForest', 'GradientBoosting']:
            model_map = {
                'DecisionTree': 'decision_tree',
                'RandomForest': 'random_forest',
                'GradientBoosting': 'gradient_boosting'
            }
            base_model = create_tree_model(model_type=model_map[model_name])
        elif model_name == 'XGBoost':
            base_model = create_tree_model(model_type='xgboost')
        else:
            logger.error(f"不支持的模型: {model_name}")
            return {}
        
        # 使用MAE作为评分标准
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=base_model.model,
            param_grid=param_grid,
            scoring=mae_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        # 拟合（使用第一个目标变量进行简化）
        grid_search.fit(data['X_train'], data['y_train'][:, 0])
        
        result = {
            'model': model_name,
            'method': 'grid_search',
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # 转回正数
            'all_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        logger.info(f"网格搜索完成:")
        logger.info(f"  最佳参数: {result['best_params']}")
        logger.info(f"  最佳MAE: {result['best_score']:.4f}")
        
        return result
    
    def random_search(self, model_name: str, data: Dict[str, Any], 
                     n_iter: int = 20) -> Dict[str, Any]:
        """
        随机搜索
        
        Parameters
        ----------
        model_name : str
            模型名称
        data : dict
            数据字典
        n_iter : int
            迭代次数
            
        Returns
        -------
        result : dict
            搜索结果
        """
        logger.info(f"开始随机搜索: {model_name} (n_iter={n_iter})")
        
        param_grid = self.get_param_grid(model_name)
        if not param_grid:
            logger.warning(f"模型 {model_name} 没有预定义的参数网格")
            return {}
        
        # 创建基础模型（同grid_search）
        if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            model_type = model_name.lower()
            if model_name == 'ElasticNet':
                model_type = 'elastic_net'
            base_model = create_linear_model(model_type=model_type)
        elif model_name in ['DecisionTree', 'RandomForest', 'GradientBoosting']:
            model_map = {
                'DecisionTree': 'decision_tree',
                'RandomForest': 'random_forest',
                'GradientBoosting': 'gradient_boosting'
            }
            base_model = create_tree_model(model_type=model_map[model_name])
        elif model_name == 'XGBoost':
            base_model = create_tree_model(model_type='xgboost')
        else:
            logger.error(f"不支持的模型: {model_name}")
            return {}
        
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        
        random_search = RandomizedSearchCV(
            estimator=base_model.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=mae_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(data['X_train'], data['y_train'][:, 0])
        
        result = {
            'model': model_name,
            'method': 'random_search',
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'all_results': pd.DataFrame(random_search.cv_results_)
        }
        
        logger.info(f"随机搜索完成:")
        logger.info(f"  最佳参数: {result['best_params']}")
        logger.info(f"  最佳MAE: {result['best_score']:.4f}")
        
        return result
    
    def tune(self, model_name: str, method: str = 'grid') -> Dict[str, Any]:
        """
        执行超参数调优
        
        Parameters
        ----------
        model_name : str
            模型名称
        method : str
            调优方法 ('grid', 'random', 'bayesian')
            
        Returns
        -------
        result : dict
            调优结果
        """
        logger.info("=" * 60)
        logger.info(f"开始超参数调优: {model_name} ({method})")
        logger.info("=" * 60)
        
        # 准备数据
        data = self.prepare_data()
        
        # 执行调优
        if method == 'grid':
            result = self.grid_search(model_name, data)
        elif method == 'random':
            result = self.random_search(model_name, data)
        elif method == 'bayesian':
            logger.warning("贝叶斯优化暂未实现，使用随机搜索代替")
            result = self.random_search(model_name, data)
        else:
            logger.error(f"未知的调优方法: {method}")
            return {}
        
        # 保存结果
        if result:
            self.save_tuning_result(result)
        
        logger.info("=" * 60)
        logger.info("超参数调优完成")
        logger.info("=" * 60)
        
        return result
    
    def save_tuning_result(self, result: Dict[str, Any]):
        """
        保存调优结果
        
        Parameters
        ----------
        result : dict
            调优结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = result['model']
        method = result['method']
        
        # 保存JSON
        json_path = os.path.join(
            self.config.result_paths['logs'],
            f'tuning_{model_name}_{method}_{timestamp}.json'
        )
        
        # 移除DataFrame，只保存可序列化的数据
        save_data = {
            'model': result['model'],
            'method': result['method'],
            'best_params': result['best_params'],
            'best_score': result['best_score'],
            'timestamp': timestamp
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果CSV
        if 'all_results' in result:
            csv_path = os.path.join(
                self.config.result_paths['logs'],
                f'tuning_{model_name}_{method}_{timestamp}.csv'
            )
            result['all_results'].to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"调优结果已保存:")
        logger.info(f"  JSON: {json_path}")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='超参数调优')
    parser.add_argument('--model', type=str, required=True,
                       help='模型名称')
    parser.add_argument('--method', type=str, default='grid',
                       choices=['grid', 'random', 'bayesian'],
                       help='调优方法（默认: grid）')
    parser.add_argument('--list', action='store_true',
                       help='列出支持调优的模型')
    
    args = parser.parse_args()
    
    # 列出支持调优的模型
    if args.list:
        print("\n支持超参数调优的模型:")
        print("  1. Ridge")
        print("  2. Lasso")
        print("  3. ElasticNet")
        print("  4. DecisionTree")
        print("  5. RandomForest")
        print("  6. GradientBoosting")
        print("  7. XGBoost")
        return
    
    # 加载配置
    config = Config()
    
    # 创建调优器
    tuner = HyperparameterTuner(config)
    
    # 执行调优
    result = tuner.tune(args.model, args.method)
    
    # 打印结果
    if result:
        print("\n" + "=" * 60)
        print("调优结果")
        print("=" * 60)
        print(f"模型: {result['model']}")
        print(f"方法: {result['method']}")
        print(f"最佳MAE: {result['best_score']:.4f}")
        print("\n最佳参数:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
        print("=" * 60)


if __name__ == '__main__':
    main()
