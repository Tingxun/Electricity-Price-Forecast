"""
模型训练脚本
用于批量训练所有模型并保存
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import (
    create_linear_model, create_tree_model, create_epf_model
)
from utils.metrics import calculate_mae, calculate_rmse, calculate_mape, calculate_smape

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    模型训练器
    负责批量训练所有模型并保存
    """
    
    def __init__(self, config: Config):
        """
        初始化训练器
        
        Parameters
        ----------
        config : Config
            项目配置对象
        """
        self.config = config
        self.results = {}
        
        # 创建保存目录
        for path in config.model_paths.values():
            os.makedirs(path, exist_ok=True)
        
        logger.info("模型训练器初始化完成")
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        准备训练数据
        从features目录加载已生成的特征数据
        
        Returns
        -------
        data_dict : dict
            包含训练集、验证集、测试集的字典
        """
        logger.info("开始准备数据...")
        
        # 从features目录加载特征数据
        engineer = FeatureEngineer()
        try:
            features_df, target_cols = engineer.load_features()
            logger.info("已从features目录加载特征数据")
        except FileNotFoundError as e:
            logger.error(f"特征数据未找到: {e}")
            logger.error("请先运行 generate_features.py 生成特征数据")
            raise
        
        # 准备特征和目标
        feature_cols = [col for col in features_df.columns if col not in target_cols + ['Date', 'Hour', '日期', '时段', 'datetime']]
        # 只选择数值型特征
        numeric_feature_cols = features_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = features_df[numeric_feature_cols].values
        y = features_df[target_cols].values
        
        # 时间序列划分
        dates = features_df['Date'].unique()
        n_dates = len(dates)
        
        # 划分比例：训练集80%，验证集10%，测试集10%
        train_end = int(n_dates * 0.8)
        val_end = int(n_dates * 0.9)
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        
        # 根据日期划分数据集
        train_mask = features_df['Date'].isin(train_dates)
        val_mask = features_df['Date'].isin(val_dates)
        test_mask = features_df['Date'].isin(test_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 特征标准化（使用训练集的统计量标准化所有数据集）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        logger.info(f"数据划分完成:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_val)} 样本")
        logger.info(f"  测试集: {len(X_test)} 样本")
        logger.info(f"  特征数: {len(numeric_feature_cols)}")
        logger.info(f"  目标数: {len(target_cols)}")
        logger.info(f"  特征标准化: 已应用 (StandardScaler)")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': numeric_feature_cols,
            'target_cols': target_cols,
            'scaler': scaler
        }
    
    def get_model_list(self, model_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        获取要训练的模型列表
        
        Parameters
        ----------
        model_names : list, optional
            指定要获取的模型名称列表，默认为None（获取所有）
            
        Returns
        -------
        model_list : list
            模型配置列表
        """
        all_models = [
            # 线性模型
            {'name': 'LinearRegression', 'type': 'linear', 'params': {'model_type': 'linear'}},
            {'name': 'Ridge', 'type': 'linear', 'params': {'model_type': 'ridge', 'alpha': 1.0}},
            {'name': 'Lasso', 'type': 'linear', 'params': {'model_type': 'lasso', 'alpha': 0.1}},
            {'name': 'ElasticNet', 'type': 'linear', 'params': {'model_type': 'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.5}},
            
            # 树模型
            {'name': 'DecisionTree', 'type': 'tree', 'params': {'model_type': 'decision_tree', 'max_depth': 10}},
            {'name': 'RandomForest', 'type': 'tree', 'params': {'model_type': 'random_forest', 'n_estimators': 100}},
            {'name': 'GradientBoosting', 'type': 'tree', 'params': {'model_type': 'gradient_boosting', 'n_estimators': 100}},
        ]
        
        # 根据model_names筛选基础模型
        if model_names is not None:
            models = [m for m in all_models if m['name'] in model_names]
        else:
            models = all_models.copy()
        
        # 尝试添加XGBoost（仅在用户指定了XGBoost或训练所有模型时检查）
        if model_names is None or 'XGBoost' in model_names:
            try:
                import xgboost
                models.append({'name': 'XGBoost', 'type': 'tree', 'params': {'model_type': 'xgboost', 'n_estimators': 100}})
            except ImportError:
                if model_names is not None and 'XGBoost' in model_names:
                    logger.warning("XGBoost未安装，跳过")
        
        # 神经网络模型（仅在用户指定了神经网络模型或训练所有模型时添加）
        neural_models = [
            {'name': 'MLP', 'type': 'neural', 'params': {'model_type': 'mlp', 'hidden_dims': [128, 64], 'epochs': 50}},
            {'name': 'LSTM', 'type': 'neural', 'params': {'model_type': 'lstm', 'hidden_dim': 64, 'num_layers': 2, 'epochs': 50}},
            {'name': 'GRU', 'type': 'neural', 'params': {'model_type': 'gru', 'hidden_dim': 64, 'num_layers': 2, 'epochs': 50}},
            {'name': 'Transformer', 'type': 'neural', 'params': {'model_type': 'transformer', 'hidden_dim': 64, 'num_layers': 2, 'epochs': 50}},
        ]
        if model_names is None:
            models.extend(neural_models)
        else:
            for nm in neural_models:
                if nm['name'] in model_names:
                    models.append(nm)
        
        # 尝试添加LEAR（仅在用户指定了LEAR或训练所有模型时检查）
        if model_names is None or 'LEAR' in model_names:
            try:
                import epftoolbox
                models.append({'name': 'LEAR', 'type': 'epf', 'params': {'model_type': 'LEAR'}})
            except ImportError:
                if model_names is not None and 'LEAR' in model_names:
                    logger.warning("epftoolbox未安装，跳过LEAR模型")
        
        return models
    
    def train_model(self, model_config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        训练单个模型
        
        Parameters
        ----------
        model_config : dict
            模型配置
        data : dict
            数据字典
            
        Returns
        -------
        result : dict
            训练结果
        """
        name = model_config['name']
        model_type = model_config['type']
        params = model_config['params']
        
        logger.info(f"\n开始训练模型: {name}")
        start_time = time.time()
        
        try:
            # 创建模型
            if model_type == 'linear':
                model = create_linear_model(**params)
                save_dir = self.config.model_paths['linear']
            elif model_type == 'tree':
                model = create_tree_model(**params)
                save_dir = self.config.model_paths['tree']
            elif model_type == 'neural':
                # 延迟导入神经网络模型
                from models import _load_neural_networks, create_neural_network
                _load_neural_networks()
                model = create_neural_network(**params)
                save_dir = self.config.model_paths['neural']
            elif model_type == 'epf':
                model = create_epf_model(**params)
                save_dir = self.config.model_paths['epftoolbox']
            else:
                raise ValueError(f"未知的模型类型: {model_type}")
            
            # 训练模型
            model.fit(data['X_train'], data['y_train'])
            
            # 验证集评估
            val_predictions = model.predict(data['X_val'])
            val_mae = calculate_mae(data['y_val'], val_predictions)
            val_rmse = calculate_rmse(data['y_val'], val_predictions)
            val_mape = calculate_mape(data['y_val'], val_predictions)
            val_smape = calculate_smape(data['y_val'], val_predictions)
            
            # 保存模型
            save_path = os.path.join(save_dir, f"{name}.pkl")
            model.save(save_path)
            
            training_time = time.time() - start_time
            
            result = {
                'name': name,
                'type': model_type,
                'status': 'success',
                'training_time': training_time,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_mape': val_mape,
                'val_smape': val_smape,
                'save_path': save_path
            }
            
            logger.info(f"模型 {name} 训练完成:")
            logger.info(f"  训练时间: {training_time:.2f}秒")
            logger.info(f"  验证集MAE: {val_mae:.4f}")
            logger.info(f"  验证集RMSE: {val_rmse:.4f}")
            logger.info(f"  验证集MAPE: {val_mape:.2f}%")
            logger.info(f"  验证集sMAPE: {val_smape:.2f}%")
            
        except Exception as e:
            logger.error(f"模型 {name} 训练失败: {str(e)}")
            result = {
                'name': name,
                'type': model_type,
                'status': 'failed',
                'error': str(e)
            }
        
        return result
    
    def train_all_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量训练所有模型
        
        Parameters
        ----------
        model_names : list, optional
            指定要训练的模型名称列表，默认为None（训练所有）
            
        Returns
        -------
        results_df : pd.DataFrame
            训练结果汇总表
        """
        logger.info("=" * 60)
        logger.info("开始批量训练模型")
        logger.info("=" * 60)
        
        # 准备数据
        data = self.prepare_data()
        
        # 获取模型列表（传入model_names以避免不必要的依赖检查）
        all_models = self.get_model_list(model_names)
        
        logger.info(f"共 {len(all_models)} 个模型需要训练")
        
        # 训练每个模型
        results = []
        for i, model_config in enumerate(all_models, 1):
            logger.info(f"\n[{i}/{len(all_models)}] 训练进度")
            result = self.train_model(model_config, data)
            results.append(result)
        
        # 生成结果表
        results_df = pd.DataFrame(results)
        
        # 保存结果
        self.save_training_report(results_df)
        
        logger.info("\n" + "=" * 60)
        logger.info("批量训练完成")
        logger.info("=" * 60)
        
        return results_df
    
    def save_training_report(self, results_df: pd.DataFrame):
        """
        保存训练报告
        
        Parameters
        ----------
        results_df : pd.DataFrame
            训练结果数据框
        """
        # 保存CSV
        report_path = os.path.join(self.config.result_paths['logs'], 'training_report.csv')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        results_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        
        # 保存JSON
        json_path = os.path.join(self.config.result_paths['logs'], 'training_report.json')
        results_df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        
        logger.info(f"训练报告已保存:")
        logger.info(f"  CSV: {report_path}")
        logger.info(f"  JSON: {json_path}")
        
        # 打印成功模型摘要
        success_models = results_df[results_df['status'] == 'success']
        if len(success_models) > 0:
            logger.info(f"\n成功训练的模型 ({len(success_models)}/{len(results_df)}):")
            summary = success_models[['name', 'val_mae', 'val_rmse', 'val_mape', 'training_time']].sort_values('val_mae')
            logger.info("\n" + str(summary))


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='训练电价预测模型')
    parser.add_argument('--models', nargs='+', default=None,
                       help='指定要训练的模型名称（默认训练所有）')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用模型')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 创建训练器
    trainer = ModelTrainer(config)
    
    # 列出模型
    if args.list:
        models = trainer.get_model_list()
        print("\n可用模型列表:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']} ({model['type']})")
        return
    
    # 训练模型
    results = trainer.train_all_models(args.models)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("训练结果汇总")
    print("=" * 80)
    print(results.to_string())


if __name__ == '__main__':
    main()
