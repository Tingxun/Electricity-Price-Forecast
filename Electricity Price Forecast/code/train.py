"""
模型训练脚本
用于批量训练所有模型并保存
支持模型特定的特征选择
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from feature_selector import FeatureSelector
from models import (
    create_linear_model, create_tree_model, create_epf_model
)
from utils.metrics import calculate_mae, calculate_rmse, calculate_smape

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
    支持模型特定的特征选择
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
        
        # 初始化特征选择器
        self.feature_selector = FeatureSelector()
        
        # 创建保存目录
        for path in config.model_paths.values():
            os.makedirs(path, exist_ok=True)
        
        logger.info("模型训练器初始化完成")
        logger.info("特征选择器已加载")
    
    def prepare_data(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        准备训练数据
        从features目录加载已生成的特征数据
        支持模型特定的特征选择
        
        Parameters
        ----------
        model_name : str, optional
            模型名称，用于选择特定特征。不指定则使用所有特征
        
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
            logger.error("请先运行: python main.py features")
            raise
        
        # 获取所有可用的数值型特征列
        all_feature_cols = [col for col in features_df.columns 
                           if col not in target_cols + ['预测日期']]
        numeric_feature_cols = features_df[all_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # 根据模型选择特征
        if model_name:
            selected_features = self.feature_selector.select_features_for_model(
                model_name, numeric_feature_cols
            )
            feature_info = self.feature_selector.get_model_feature_info(model_name)
            logger.info(f"模型 {model_name} 使用 {len(selected_features)} 个选定特征")
        else:
            selected_features = numeric_feature_cols
            feature_info = {'normalize': True}  # 默认需要标准化
            logger.info(f"使用所有 {len(selected_features)} 个特征")
        
        # 准备特征矩阵
        X = features_df[selected_features].values
        y = features_df[target_cols].values
        
        # 时间序列划分（基于预测日期）
        dates = features_df['预测日期'].values
        n_dates = len(dates)
        
        # 划分比例：训练集80%，验证集10%，测试集10%
        train_end = int(n_dates * 0.8)
        val_end = int(n_dates * 0.9)
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        
        # 根据日期划分数据集
        train_mask = features_df['预测日期'].isin(train_dates)
        val_mask = features_df['预测日期'].isin(val_dates)
        test_mask = features_df['预测日期'].isin(test_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 根据模型配置决定是否标准化
        scaler = None
        if feature_info.get('normalize', True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            logger.info(f"  特征标准化: 已应用 (StandardScaler)")
        else:
            logger.info(f"  特征标准化: 已跳过")
        
        logger.info(f"数据划分完成:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_val)} 样本")
        logger.info(f"  测试集: {len(X_test)} 样本")
        logger.info(f"  特征数: {len(selected_features)}")
        logger.info(f"  目标数: {len(target_cols)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': selected_features,
            'target_cols': target_cols,
            'scaler': scaler,
            'model_name': model_name
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
    
    def train_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        训练单个模型
        为每个模型准备特定的特征数据
        
        Parameters
        ----------
        model_config : dict
            模型配置
            
        Returns
        -------
        result : dict
            训练结果
        """
        name = model_config['name']
        model_type = model_config['type']
        params = model_config['params']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练模型: {name}")
        logger.info(f"{'='*60}")
        start_time = time.time()
        
        # 为当前模型准备数据（使用模型特定的特征）
        data = self.prepare_data(model_name=name)
        
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
                'val_smape': val_smape,
                'save_path': save_path
            }
            
            logger.info(f"模型 {name} 训练完成:")
            logger.info(f"  训练时间: {training_time:.2f}秒")
            logger.info(f"  验证集MAE: {val_mae:.4f}")
            logger.info(f"  验证集RMSE: {val_rmse:.4f}")
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
        每个模型使用其特定的特征集
        
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
        logger.info("注意：每个模型将使用其特定的特征集")
        logger.info("=" * 60)
        
        # 获取模型列表（传入model_names以避免不必要的依赖检查）
        all_models = self.get_model_list(model_names)
        
        logger.info(f"共 {len(all_models)} 个模型需要训练")
        
        # 训练每个模型（每个模型独立准备数据以使用特定特征）
        results = []
        for i, model_config in enumerate(all_models, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(all_models)}] 总体进度")
            logger.info(f"{'='*60}")
            result = self.train_model(model_config)
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
            summary = success_models[['name', 'val_mae', 'val_rmse', 'val_smape', 'training_time']].sort_values('val_mae')
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
