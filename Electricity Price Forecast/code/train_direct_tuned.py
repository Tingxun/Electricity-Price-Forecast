"""
Direct多步预测模型训练脚本（带独立超参数调优）

为每个时点模型进行独立的随机超参数搜索
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from feature_engineering_direct import DirectFeatureEngineer
from utils.metrics import calculate_mae, calculate_rmse, calculate_smape

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_direct_tuned.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# LightGBM超参数搜索空间
LGBM_PARAM_SPACE = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'num_leaves': [15, 31, 63, 127, 255],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
    'min_child_samples': [5, 10, 20, 30, 50]
}

# XGBoost超参数搜索空间
XGB_PARAM_SPACE = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
    'min_child_weight': [1, 3, 5, 7]
}

# Random Forest超参数搜索空间
RF_PARAM_SPACE = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}


class DirectModelTuner:
    """
    Direct多步预测模型调优器
    
    为每个时点模型独立进行超参数搜索
    """
    
    def __init__(self, config: Config, model_type: str = 'lightgbm', 
                 n_iter: int = 20, cv_folds: int = 3):
        """
        初始化调优器
        
        Parameters
        ----------
        config : Config
            项目配置
        model_type : str
            模型类型
        n_iter : int
            每个时点的随机搜索次数
        cv_folds : int
            交叉验证折数
        """
        self.config = config
        self.model_type = model_type
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        
        # 创建保存目录
        self.model_dir = config.model_paths['tree'] / 'direct_tuned' / model_type
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 获取参数空间
        if model_type == 'lightgbm':
            self.param_space = LGBM_PARAM_SPACE
        elif model_type == 'xgboost':
            self.param_space = XGB_PARAM_SPACE
        elif model_type == 'random_forest':
            self.param_space = RF_PARAM_SPACE
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        logger.info(f"Direct模型调优器初始化完成")
        logger.info(f"模型类型: {model_type}")
        logger.info(f"搜索迭代次数: {n_iter}")
        logger.info(f"交叉验证折数: {cv_folds}")
    
    def prepare_hourly_data(self, hour: int) -> Dict[str, Any]:
        """准备指定小时的训练数据"""
        engineer = DirectFeatureEngineer()
        features_df, target_col = engineer.load_features(hour)
        
        feature_cols = [c for c in features_df.columns 
                       if c not in [target_col, '预测日期']]
        
        # 使用DataFrame保留特征名，避免UserWarning
        X = features_df[feature_cols]
        y = features_df[target_col].values
        dates = features_df['预测日期'].values
        
        # 时间序列划分：训练集90%，测试集10%
        n_dates = len(dates)
        train_end = int(n_dates * 0.9)
        
        train_dates = dates[:train_end]
        test_dates = dates[train_end:]
        
        train_mask = features_df['预测日期'].isin(train_dates)
        test_mask = features_df['预测日期'].isin(test_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': feature_cols,
            'target_col': target_col
        }
    
    def create_model(self, params: Dict):
        """根据参数创建模型"""
        if self.model_type == 'lightgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(verbose=-1, **params)
        elif self.model_type == 'xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
    
    def tune_single_hour(self, hour: int) -> Dict[str, Any]:
        """
        为单个时点进行超参数调优
        
        Parameters
        ----------
        hour : int
            预测步长（0-23）
            
        Returns
        -------
        result : dict
            调优结果，包含最优参数和性能
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"调优 {hour:02d}:00 步长模型")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # 准备数据
        data = self.prepare_hourly_data(hour)
        logger.info(f"训练集: {len(data['X_train'])} 样本")
        logger.info(f"测试集: {len(data['X_test'])} 样本")
        logger.info(f"特征数: {len(data['feature_cols'])}")
        
        # 生成随机参数组合
        param_list = list(ParameterSampler(self.param_space, n_iter=self.n_iter, random_state=42+hour))
        
        best_score = float('inf')
        best_params = None
        best_model = None
        search_results = []
        
        logger.info(f"开始随机搜索 {self.n_iter} 组参数（使用训练集交叉验证）...")
        
        # 在训练集上进行时间序列交叉验证
        X_train_full = data['X_train']
        y_train_full = data['y_train']
        n_train = len(X_train_full)
        
        for i, params in enumerate(param_list, 1):
            try:
                # 时间序列交叉验证（3折）
                cv_scores = []
                fold_sizes = n_train // 3
                
                for fold in range(3):
                    # 划分训练集和验证集
                    val_start = fold * fold_sizes
                    val_end = (fold + 1) * fold_sizes if fold < 2 else n_train
                    
                    X_tr = pd.concat([X_train_full.iloc[:val_start], X_train_full.iloc[val_end:]], ignore_index=True)
                    y_tr = np.concatenate([y_train_full[:val_start], y_train_full[val_end:]])
                    X_val = X_train_full.iloc[val_start:val_end]
                    y_val = y_train_full[val_start:val_end]
                    
                    # 训练和评估
                    model = self.create_model(params)
                    model.fit(X_tr, y_tr)
                    val_pred = model.predict(X_val)
                    cv_scores.append(calculate_mae(y_val, val_pred))
                
                # 平均交叉验证得分
                cv_mae = np.mean(cv_scores)
                
                search_results.append({
                    'iteration': i,
                    'params': params,
                    'cv_mae': cv_mae
                })
                
                # 更新最优参数
                if cv_mae < best_score:
                    best_score = cv_mae
                    best_params = params.copy()
                    logger.info(f"  [{i}/{self.n_iter}] 新最优: CV_MAE={cv_mae:.4f}, 参数={params}")
                else:
                    if i % 5 == 0:
                        logger.info(f"  [{i}/{self.n_iter}] CV_MAE={cv_mae:.4f}")
                
            except Exception as e:
                logger.warning(f"  [{i}/{self.n_iter}] 参数失败: {str(e)}")
                continue
        
        tuning_time = time.time() - start_time
        
        # 使用最优参数重新训练（使用全部训练数据）
        logger.info(f"\n使用最优参数在全部训练数据上重新训练...")
        final_model = self.create_model(best_params)
        final_model.fit(X_train_full, y_train_full)
        
        # 保存模型
        import joblib
        save_path = self.model_dir / f'model_H{hour:02d}.pkl'
        joblib.dump(final_model, str(save_path))
        
        # 保存最优参数
        params_path = self.model_dir / f'params_H{hour:02d}.json'
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        
        result = {
            'hour': hour,
            'status': 'success',
            'tuning_time': tuning_time,
            'best_params': best_params,
            'best_cv_mae': best_score,
            'n_searched': len(search_results),
            'save_path': str(save_path)
        }
        
        logger.info(f"\n{hour:02d}:00 调优完成:")
        logger.info(f"  调优时间: {tuning_time:.2f}秒")
        logger.info(f"  最优CV_MAE: {best_score:.4f}")
        logger.info(f"  最优参数: {best_params}")
        
        return result
    
    def tune_all_models(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        """
        为所有时点进行超参数调优
        
        Parameters
        ----------
        hours : list, optional
            指定要调优的步长列表
            
        Returns
        -------
        results_df : pd.DataFrame
            调优结果汇总
        """
        if hours is None:
            hours = list(range(24))
        
        logger.info("=" * 60)
        logger.info(f"开始Direct多步预测超参数调优 - {self.model_type}")
        logger.info("=" * 60)
        logger.info(f"共 {len(hours)} 个预测步长需要调优")
        logger.info(f"每步长搜索 {self.n_iter} 组参数")
        
        # 调优每个步长
        results = []
        for i, hour in enumerate(hours, 1):
            logger.info(f"\n[{i}/{len(hours)}] 总体进度")
            result = self.tune_single_hour(hour)
            results.append(result)
        
        # 生成结果表
        results_df = pd.DataFrame([r for r in results if r['status'] == 'success'])
        
        # 保存结果
        self.save_tuning_report(results_df)
        
        logger.info("\n" + "=" * 60)
        logger.info("Direct多步预测超参数调优完成")
        logger.info("=" * 60)
        
        if len(results_df) > 0:
            logger.info(f"\n调优摘要:")
            logger.info(f"  平均最优CV_MAE: {results_df['best_cv_mae'].mean():.4f}")
            logger.info(f"  总调优时间: {results_df['tuning_time'].sum():.2f}秒")
        
        return results_df
    
    def save_tuning_report(self, results_df: pd.DataFrame):
        """保存调优报告"""
        # 保存CSV
        report_path = self.model_dir / 'tuning_report.csv'
        results_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        
        # 保存JSON（包含详细参数）
        json_path = self.model_dir / 'tuning_report.json'
        results_df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        
        logger.info(f"调优报告已保存到: {self.model_dir}")
    
    def evaluate_tuned_models(self, hours: List[int] = None):
        """
        评估调优后的模型
        
        Parameters
        ----------
        hours : list
            要评估的步长
        """
        if hours is None:
            hours = list(range(24))
        
        logger.info("=" * 60)
        logger.info("评估调优后的Direct多步预测模型")
        logger.info("=" * 60)
        
        import joblib
        engineer = DirectFeatureEngineer()
        
        all_predictions = []
        all_actuals = []
        
        for hour in hours:
            # 加载模型
            model_path = self.model_dir / f'model_H{hour:02d}.pkl'
            if not model_path.exists():
                continue
            
            model = joblib.load(model_path)
            
            # 加载测试数据
            features_df, target_col = engineer.load_features(hour)
            
            dates = features_df['预测日期'].values
            test_start = int(len(dates) * 0.9)
            test_dates = dates[test_start:]
            test_mask = features_df['预测日期'].isin(test_dates)
            
            test_df = features_df[test_mask]
            feature_cols = [c for c in test_df.columns if c not in [target_col, '预测日期']]
            
            # 使用DataFrame保留特征名，避免UserWarning
            X_test = test_df[feature_cols]
            y_test = test_df[target_col].values
            
            # 预测
            y_pred = model.predict(X_test)
            
            all_predictions.append(y_pred)
            all_actuals.append(y_test)
            
            # 计算指标
            mae = calculate_mae(y_test, y_pred)
            smape = calculate_smape(y_test, y_pred)
            
            logger.info(f"{hour:02d}:00 - MAE: {mae:.4f}, sMAPE: {smape:.2f}%")
        
        # 整体评估
        all_pred = np.column_stack(all_predictions)
        all_actual = np.column_stack(all_actuals)
        
        overall_mae = calculate_mae(all_actual, all_pred)
        overall_smape = calculate_smape(all_actual, all_pred)
        
        logger.info("\n" + "=" * 60)
        logger.info("整体评估结果")
        logger.info("=" * 60)
        logger.info(f"平均MAE: {overall_mae:.4f}")
        logger.info(f"平均sMAPE: {overall_smape:.2f}%")
        
        return overall_mae, overall_smape


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Direct多步预测模型超参数调优')
    parser.add_argument('--model', type=str, default='lightgbm',
                       choices=['lightgbm', 'xgboost', 'random_forest'],
                       help='模型类型')
    parser.add_argument('--hours', type=int, nargs='+', default=None,
                       help='指定调优的步长，如：--hours 0 8 12 18')
    parser.add_argument('--n_iter', type=int, default=20,
                       help='每步长的随机搜索次数')
    parser.add_argument('--evaluate', action='store_true',
                       help='调优后评估模型')
    
    args = parser.parse_args()
    
    # 检查特征数据是否存在
    feature_path = Config().get_data_path('features') / 'direct'
    if not feature_path.exists():
        logger.error(f"Direct特征不存在: {feature_path}")
        logger.error("请先运行: python code/feature_engineering_direct.py")
        return
    
    # 检查是否需要重新生成特征（含气象数据）
    sample_file = feature_path / 'features_H00.csv'
    if sample_file.exists():
        sample_df = pd.read_csv(sample_file)
        if len(sample_df.columns) < 100:
            logger.warning("特征数据可能不包含气象数据，建议重新生成特征")
    
    # 调优模型
    config = Config()
    tuner = DirectModelTuner(config, model_type=args.model, n_iter=args.n_iter)
    results = tuner.tune_all_models(hours=args.hours)
    
    # 评估模型
    if args.evaluate:
        tuner.evaluate_tuned_models(hours=args.hours)
    
    print("\n调优完成！")


if __name__ == '__main__':
    main()
