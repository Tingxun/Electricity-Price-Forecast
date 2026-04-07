"""
超参数优化脚本
用于优化模型超参数
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

# 忽略所有警告
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from feature_engineering import FeatureEngineer
from feature_selector import FeatureSelector
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
        
        # 初始化特征选择器
        self.feature_selector = FeatureSelector()
        
        # 创建保存目录
        os.makedirs(config.result_paths['logs'], exist_ok=True)
        
        logger.info("超参数调优器初始化完成")
    
    def prepare_data(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        准备数据（从features目录加载已生成的特征）
        
        Parameters
        ----------
        model_name : str, optional
            模型名称，用于选择特定特征
            
        Returns
        -------
        data_dict : dict
            数据字典
        """
        logger.info("开始准备数据...")
        
        # 从features目录加载特征数据
        engineer = FeatureEngineer()
        try:
            features_df, target_cols = engineer.load_features()
            logger.info("已从features目录加载特征数据")
        except FileNotFoundError as e:
            logger.error(f"特征数据未找到: {e}")
            logger.error("请先运行: python main.py features")
            raise
        
        # 获取所有可用的数值型特征列（排除目标变量和日期列）
        all_feature_cols = [col for col in features_df.columns 
                           if col not in target_cols + ['预测日期']]
        numeric_feature_cols = features_df[all_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # 根据模型选择特征（如果指定了模型名称）
        if model_name:
            selected_features = self.feature_selector.select_features_for_model(
                model_name, numeric_feature_cols
            )
            logger.info(f"模型 {model_name} 使用 {len(selected_features)} 个选定特征")
        else:
            # 默认使用所有数值特征
            selected_features = numeric_feature_cols
            logger.info(f"使用所有 {len(selected_features)} 个特征")
        
        # 准备特征矩阵
        X = features_df[selected_features].values
        y = features_df[target_cols].values
        
        # 时间序列划分（基于预测日期）
        dates = features_df['预测日期'].values
        n_dates = len(dates)
        
        train_end = int(n_dates * 0.8)
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:]
        
        train_mask = features_df['预测日期'].isin(train_dates)
        val_mask = features_df['预测日期'].isin(val_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        logger.info(f"数据准备完成:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_val)} 样本")
        logger.info(f"  特征数: {len(selected_features)}")
        logger.info(f"  目标数: {len(target_cols)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'feature_cols': selected_features
        }
    
    def get_param_grid(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型的参数搜索空间
        
        由于模型被MultiOutputRegressor包装，所有参数都需要加上estimator__前缀
        
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
                'estimator__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'estimator__alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'ElasticNet': {
                'estimator__alpha': [0.001, 0.01, 0.1, 1.0],
                'estimator__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'RandomForest': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2]
            },
            'GradientBoosting': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 0.2],
                'estimator__max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 0.2],
                'estimator__max_depth': [3, 5, 7],
                'estimator__subsample': [0.8, 1.0]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def get_bayesian_param_space(self, model_name: str, trial) -> Dict[str, Any]:
        """
        获取贝叶斯优化的参数搜索空间（用于optuna）
        
        Parameters
        ----------
        model_name : str
            模型名称
        trial : optuna.Trial
            optuna试验对象
            
        Returns
        -------
        params : dict
            参数字典
        """
        if model_name == 'Ridge':
            return {
                'estimator__alpha': trial.suggest_float('estimator__alpha', 0.001, 100.0, log=True)
            }
        elif model_name == 'Lasso':
            return {
                'estimator__alpha': trial.suggest_float('estimator__alpha', 0.0001, 10.0, log=True)
            }
        elif model_name == 'ElasticNet':
            return {
                'estimator__alpha': trial.suggest_float('estimator__alpha', 0.0001, 10.0, log=True),
                'estimator__l1_ratio': trial.suggest_float('estimator__l1_ratio', 0.0, 1.0)
            }
        elif model_name == 'RandomForest':
            return {
                'estimator__n_estimators': trial.suggest_int('estimator__n_estimators', 50, 500),
                'estimator__max_depth': trial.suggest_int('estimator__max_depth', 3, 30),
                'estimator__min_samples_split': trial.suggest_int('estimator__min_samples_split', 2, 20),
                'estimator__min_samples_leaf': trial.suggest_int('estimator__min_samples_leaf', 1, 10)
            }
        elif model_name == 'GradientBoosting':
            return {
                'estimator__n_estimators': trial.suggest_int('estimator__n_estimators', 50, 500),
                'estimator__learning_rate': trial.suggest_float('estimator__learning_rate', 0.001, 0.5, log=True),
                'estimator__max_depth': trial.suggest_int('estimator__max_depth', 2, 10),
                'estimator__subsample': trial.suggest_float('estimator__subsample', 0.5, 1.0)
            }
        elif model_name == 'XGBoost':
            return {
                'estimator__n_estimators': trial.suggest_int('estimator__n_estimators', 50, 500),
                'estimator__learning_rate': trial.suggest_float('estimator__learning_rate', 0.001, 0.5, log=True),
                'estimator__max_depth': trial.suggest_int('estimator__max_depth', 2, 10),
                'estimator__subsample': trial.suggest_float('estimator__subsample', 0.5, 1.0),
                'estimator__colsample_bytree': trial.suggest_float('estimator__colsample_bytree', 0.5, 1.0)
            }
        else:
            return {}
    
    def bayesian_optimization(self, model_name: str, data: Dict[str, Any],
                             n_trials: int = 50, timeout: int = 600) -> Dict[str, Any]:
        """
        贝叶斯优化（使用Optuna的TPE算法）
        
        Parameters
        ----------
        model_name : str
            模型名称
        data : dict
            数据字典
        n_trials : int
            试验次数
        timeout : int
            超时时间（秒）
            
        Returns
        -------
        result : dict
            优化结果
        """
        
        logger.info(f"开始贝叶斯优化: {model_name} (n_trials={n_trials})")
        
        # 创建基础模型
        if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            model_type = model_name.lower()
            if model_name == 'ElasticNet':
                model_type = 'elastic_net'
            base_model = create_linear_model(model_type=model_type)
        elif model_name in ['RandomForest', 'GradientBoosting']:
            model_map = {
                'RandomForest': 'random_forest',
                'GradientBoosting': 'gradient_boosting'
            }
            base_model = create_tree_model(model_type=model_map[model_name])
        elif model_name == 'XGBoost':
            base_model = create_tree_model(model_type='xgboost')
        else:
            logger.error(f"不支持的模型: {model_name}")
            return {}
        
        # 定义目标函数
        def objective(trial):
            # 获取参数
            params = self.get_bayesian_param_space(model_name, trial)
            if not params:
                return float('inf')
            
            # 创建模型并设置参数
            model = base_model.model
            model.set_params(**params)
            
            # 使用交叉验证评估
            try:
                scores = cross_val_score(
                    model, data['X_train'], data['y_train'],
                    scoring='neg_mean_absolute_error',
                    cv=3,
                    n_jobs=-1
                )
                # 返回负MAE（optuna默认最小化）
                return -scores.mean()
            except Exception as e:
                logger.warning(f"试验失败: {e}")
                return float('inf')
        
        # 创建study并优化
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # 获取最佳结果
        best_params = study.best_params
        best_score = study.best_value
        
        result = {
            'model': model_name,
            'method': 'bayesian',
            'best_params': best_params,
            'best_score': best_score,
            'all_results': study.trials_dataframe(),
            'n_trials': len(study.trials),
            'study': study
        }
        
        logger.info(f"贝叶斯优化完成:")
        logger.info(f"  完成试验数: {result['n_trials']}")
        logger.info(f"  最佳MAE: {best_score:.4f}")
        logger.info(f"  最佳参数: {best_params}")
        
        return result
    
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
        elif model_name in ['RandomForest', 'GradientBoosting']:
            model_map = {
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
        
        # 拟合（使用完整的多输出y）
        grid_search.fit(data['X_train'], data['y_train'])
        
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
        elif model_name in ['RandomForest', 'GradientBoosting']:
            model_map = {
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
        
        random_search.fit(data['X_train'], data['y_train'])
        
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
        
        # 准备数据（传入模型名称以使用feature_config.yaml中的特征配置）
        data = self.prepare_data(model_name)
        
        # 执行调优
        if method == 'grid':
            result = self.grid_search(model_name, data)
        elif method == 'random':
            result = self.random_search(model_name, data)
        elif method == 'bayesian':
            result = self.bayesian_optimization(model_name, data)
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
