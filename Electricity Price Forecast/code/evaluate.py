"""
模型评估脚本
用于在测试集上评估模型性能并生成报告
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
import pickle

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from feature_engineering import FeatureEngineer
from feature_selector import FeatureSelector
from models import BaseModel
from utils.metrics import (
    calculate_mae, calculate_rmse, 
    calculate_smape, calculate_r2, calculate_mse
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型评估器
    负责在测试集上评估模型性能
    """
    
    def __init__(self, config: Config):
        """
        初始化评估器
        
        Parameters
        ----------
        config : Config
            项目配置对象
        """
        self.config = config
        self.results = {}
        
        # 初始化特征选择器
        self.feature_selector = FeatureSelector()
        
        # 创建结果目录
        os.makedirs(config.result_paths['predictions'], exist_ok=True)
        os.makedirs(config.result_paths['figures'], exist_ok=True)
        os.makedirs(config.result_paths['logs'], exist_ok=True)
        
        logger.info("模型评估器初始化完成")
    
    def prepare_data(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        准备评估数据（从features目录加载已生成的特征）
        
        Parameters
        ----------
        model_name : str, optional
            模型名称，用于选择特定特征
            
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
            logger.info(f"模型 {model_name} 使用 {len(selected_features)} 个选定特征")
        else:
            selected_features = numeric_feature_cols
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
        
        # 保存测试集日期信息用于后续分析
        test_dates_info = features_df[test_mask][['预测日期']].copy()
        
        logger.info(f"数据准备完成:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_val)} 样本")
        logger.info(f"  测试集: {len(X_test)} 样本")
        logger.info(f"  测试日期范围: {test_dates[0]} 至 {test_dates[-1]}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': selected_features,
            'target_cols': target_cols,
            'test_dates_info': test_dates_info
        }
    
    def load_model(self, model_path: str) -> Optional[BaseModel]:
        """
        加载模型
        
        Parameters
        ----------
        model_path : str
            模型文件路径
            
        Returns
        -------
        model : BaseModel or None
            加载的模型，失败返回None
        """
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # 创建模型实例并加载
            model = data['model']
            logger.info(f"模型加载成功: {model_path}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败 {model_path}: {str(e)}")
            return None
    
    def evaluate_model(self, model_name: str, model: BaseModel, 
                      data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个模型
        
        Parameters
        ----------
        model_name : str
            模型名称
        model : BaseModel
            模型实例
        data : dict
            数据字典
            
        Returns
        -------
        result : dict
            评估结果
        """
        logger.info(f"\n开始评估模型: {model_name}")
        start_time = time.time()
        
        try:
            # 测试集预测
            X_test = data['X_test']
            y_test = data['y_test']
            
            predictions = model.predict(X_test)
            
            # 计算评估指标
            mae = calculate_mae(y_test, predictions)
            mse = calculate_mse(y_test, predictions)
            rmse = calculate_rmse(y_test, predictions)
            smape = calculate_smape(y_test, predictions)
            r2 = calculate_r2(y_test, predictions)
            
            # 计算每个小时的指标
            hourly_mae = []
            hourly_smape = []
            for h in range(y_test.shape[1]):
                hourly_mae.append(calculate_mae(y_test[:, h], predictions[:, h]))
                hourly_smape.append(calculate_smape(y_test[:, h], predictions[:, h]))
            
            eval_time = time.time() - start_time
            
            result = {
                'name': model_name,
                'status': 'success',
                'eval_time': eval_time,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'smape': smape,
                'r2': r2,
                'hourly_mae': hourly_mae,
                'hourly_smape': hourly_smape,
                'predictions': predictions
            }
            
            logger.info(f"模型 {model_name} 评估完成:")
            logger.info(f"  评估时间: {eval_time:.2f}秒")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  sMAPE: {smape:.2f}%")
            logger.info(f"  R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"模型 {model_name} 评估失败: {str(e)}")
            result = {
                'name': model_name,
                'status': 'failed',
                'error': str(e)
            }
        
        return result
    
    def find_saved_models(self) -> List[Dict[str, str]]:
        """
        查找所有保存的模型
        
        Returns
        -------
        model_list : list
            模型路径列表
        """
        models = []
        
        for model_type, model_dir in self.config.model_paths.items():
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pkl'):
                        model_path = os.path.join(model_dir, file)
                        model_name = file.replace('.pkl', '')
                        models.append({
                            'name': model_name,
                            'type': model_type,
                            'path': model_path
                        })
        
        return models
    
    def evaluate_all_models(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量评估所有模型
        
        Parameters
        ----------
        model_names : list, optional
            指定要评估的模型名称列表，默认为None（评估所有）
            
        Returns
        -------
        results_df : pd.DataFrame
            评估结果汇总表
        """
        logger.info("=" * 60)
        logger.info("开始批量评估模型")
        logger.info("=" * 60)
        
        # 准备数据
        data = self.prepare_data()
        
        # 查找保存的模型
        saved_models = self.find_saved_models()
        
        if len(saved_models) == 0:
            logger.error("未找到任何保存的模型，请先运行训练脚本")
            return pd.DataFrame()
        
        # 筛选指定模型
        if model_names is not None:
            saved_models = [m for m in saved_models if m['name'] in model_names]
        
        logger.info(f"共 {len(saved_models)} 个模型需要评估")
        
        # 评估每个模型
        results = []
        all_predictions = {}
        
        for i, model_info in enumerate(saved_models, 1):
            logger.info(f"\n[{i}/{len(saved_models)}] 评估进度")
            
            # 加载模型
            model = self.load_model(model_info['path'])
            if model is None:
                continue
            
            # 评估模型
            result = self.evaluate_model(model_info['name'], model, data)
            results.append(result)
            
            # 保存预测结果
            if result['status'] == 'success':
                all_predictions[model_info['name']] = result['predictions']
        
        # 生成结果表
        results_df = pd.DataFrame([r for r in results if r['status'] == 'success'])
        
        # 保存结果
        self.save_evaluation_report(results_df, all_predictions, data)
        
        logger.info("\n" + "=" * 60)
        logger.info("批量评估完成")
        logger.info("=" * 60)
        
        return results_df
    
    def save_evaluation_report(self, results_df: pd.DataFrame, 
                              predictions: Dict[str, np.ndarray],
                              data: Dict[str, Any]):
        """
        保存评估报告
        
        Parameters
        ----------
        results_df : pd.DataFrame
            评估结果数据框
        predictions : dict
            各模型的预测结果
        data : dict
            数据字典
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存评估指标CSV
        metrics_path = os.path.join(self.config.result_paths['logs'], 
                                    f'evaluation_metrics_{timestamp}.csv')
        metrics_df = results_df[['name', 'mae', 'mse', 'rmse', 'smape', 'r2', 'eval_time']].copy()
        metrics_df = metrics_df.sort_values('mae')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        logger.info(f"评估指标已保存: {metrics_path}")
        
        # 2. 保存预测结果
        for model_name, preds in predictions.items():
            pred_path = os.path.join(self.config.result_paths['predictions'], 
                                    f'{model_name}_predictions_{timestamp}.csv')
            
            # 构建预测结果DataFrame
            pred_df = data['test_dates_info'].copy()
            for i, col in enumerate(data['target_cols']):
                pred_df[f'{col}_true'] = data['y_test'][:, i]
                pred_df[f'{col}_pred'] = preds[:, i]
            
            pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"预测结果已保存到: {self.config.result_paths['predictions']}")
        
        # 3. 生成详细报告
        report_path = os.path.join(self.config.result_paths['logs'], 
                                   f'evaluation_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("模型评估报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体性能排名
            f.write("【总体性能排名】\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'排名':<6}{'模型名称':<20}{'MAE':<12}{'RMSE':<12}{'sMAPE(%)':<12}{'R²':<10}\n")
            f.write("-" * 80 + "\n")
            
            for i, row in metrics_df.iterrows():
                f.write(f"{i+1:<6}{row['name']:<20}{row['mae']:<12.4f}{row['rmse']:<12.4f}"
                       f"{row['smape']:<12.2f}{row['r2']:<10.4f}\n")
            
            f.write("\n")
            
            # 每个模型的详细指标
            f.write("【详细评估指标】\n")
            f.write("-" * 80 + "\n")
            
            for _, row in results_df.iterrows():
                f.write(f"\n模型: {row['name']}\n")
                f.write(f"  MAE:  {row['mae']:.4f}\n")
                f.write(f"  MSE:  {row['mse']:.4f}\n")
                f.write(f"  RMSE: {row['rmse']:.4f}\n")
                f.write(f"  sMAPE:{row['smape']:.2f}%\n")
                f.write(f"  R²:   {row['r2']:.4f}\n")
                
                # 每小时MAE
                if 'hourly_mae' in row:
                    f.write(f"  每小时MAE: {row['hourly_mae']}\n")
        
        logger.info(f"详细报告已保存: {report_path}")
        
        # 4. 打印摘要
        logger.info("\n" + "=" * 80)
        logger.info("评估结果摘要")
        logger.info("=" * 80)
        logger.info("\n性能排名（按MAE排序）:")
        logger.info("\n" + str(metrics_df.to_string(index=False)))
        
        # 最佳模型
        if len(metrics_df) > 0:
            best_model = metrics_df.iloc[0]
            logger.info(f"\n最佳模型: {best_model['name']}")
            logger.info(f"  MAE: {best_model['mae']:.4f}")
            logger.info(f"  sMAPE: {best_model['smape']:.2f}%")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='评估电价预测模型')
    parser.add_argument('--models', nargs='+', default=None,
                       help='指定要评估的模型名称（默认评估所有）')
    parser.add_argument('--list', action='store_true',
                       help='列出所有已保存的模型')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 创建评估器
    evaluator = ModelEvaluator(config)
    
    # 列出模型
    if args.list:
        models = evaluator.find_saved_models()
        print("\n已保存的模型列表:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']} ({model['type']}) - {model['path']}")
        return
    
    # 评估模型
    results = evaluator.evaluate_all_models(args.models)
    
    # 打印结果
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("评估结果汇总")
        print("=" * 80)
        print(results[['name', 'mae', 'rmse', 'smape', 'r2']].to_string(index=False))


if __name__ == '__main__':
    main()
