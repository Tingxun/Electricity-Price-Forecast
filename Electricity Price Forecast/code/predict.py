"""
预测脚本
用于使用训练好的模型进行未来电价预测
"""

import os
import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Predictor:
    """
    预测器类
    负责加载模型并进行预测
    """
    
    def __init__(self, config: Config):
        """
        初始化预测器
        
        Parameters
        ----------
        config : Config
            项目配置对象
        """
        self.config = config
        self.model = None
        self.model_name = None
        
        logger.info("预测器初始化完成")
    
    def load_model(self, model_name: str) -> bool:
        """
        加载指定模型
        
        Parameters
        ----------
        model_name : str
            模型名称
            
        Returns
        -------
        success : bool
            是否加载成功
        """
        # 在各个模型目录中查找
        for model_type, model_dir in self.config.model_paths.items():
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    self.model = data['model']
                    self.model_name = model_name
                    
                    logger.info(f"模型加载成功: {model_name} ({model_type})")
                    return True
                except Exception as e:
                    logger.error(f"模型加载失败 {model_path}: {str(e)}")
                    continue
        
        logger.error(f"未找到模型: {model_name}")
        return False
    
    def prepare_prediction_data(self, target_date: str) -> np.ndarray:
        """
        准备预测数据
        
        Parameters
        ----------
        target_date : str
            目标预测日期 (格式: YYYY-MM-DD)
            
        Returns
        -------
        X : np.ndarray
            特征矩阵
        """
        logger.info(f"准备预测数据，目标日期: {target_date}")
        
        # 加载数据
        loader = DataLoader(self.config)
        df = loader.load_data()
        
        # 特征工程
        engineer = FeatureEngineer()
        features_df, target_cols = engineer.create_all_features(df)
        
        # 获取特征列
        feature_cols = [col for col in features_df.columns if col not in target_cols + ['Date', 'Hour']]
        
        # 找到目标日期的数据
        target_data = features_df[features_df['Date'] == target_date]
        
        if len(target_data) == 0:
            # 如果目标日期没有数据，使用最后可用的数据
            logger.warning(f"目标日期 {target_date} 无数据，使用最新可用数据")
            target_data = features_df.iloc[-24:].copy()
        
        X = target_data[feature_cols].values
        
        logger.info(f"预测数据准备完成: {X.shape}")
        return X
    
    def predict(self, model_name: str, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        进行预测
        
        Parameters
        ----------
        model_name : str
            模型名称
        target_date : str, optional
            目标预测日期，默认为明天
            
        Returns
        -------
        result : dict
            预测结果
        """
        # 加载模型
        if not self.load_model(model_name):
            return {'status': 'error', 'message': f'模型 {model_name} 加载失败'}
        
        # 确定目标日期
        if target_date is None:
            target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"使用模型 {model_name} 预测 {target_date} 的电价")
        
        # 准备数据
        X = self.prepare_prediction_data(target_date)
        
        # 进行预测
        try:
            predictions = self.model.predict(X)
            
            # 确保预测结果是24小时
            if predictions.shape[0] >= 24:
                predictions = predictions[:24]
            else:
                # 如果不足24小时，复制最后一小时的数据
                pad_length = 24 - predictions.shape[0]
                last_values = predictions[-1:] if len(predictions.shape) == 1 else predictions[-1:, :]
                predictions = np.vstack([predictions] + [last_values] * pad_length)
            
            # 如果是多输出模型，取第一列（假设是价格）
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions_24h = predictions[:, 0]
            else:
                predictions_24h = predictions.flatten()[:24]
            
            # 构建结果
            hours = list(range(24))
            result = {
                'status': 'success',
                'model': model_name,
                'target_date': target_date,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': {
                    'hours': hours,
                    'prices': predictions_24h.tolist()
                },
                'statistics': {
                    'min_price': float(np.min(predictions_24h)),
                    'max_price': float(np.max(predictions_24h)),
                    'mean_price': float(np.mean(predictions_24h)),
                    'std_price': float(np.std(predictions_24h))
                }
            }
            
            logger.info(f"预测完成:")
            logger.info(f"  最低价格: {result['statistics']['min_price']:.2f}")
            logger.info(f"  最高价格: {result['statistics']['max_price']:.2f}")
            logger.info(f"  平均价格: {result['statistics']['mean_price']:.2f}")
            
            # 保存预测结果
            self.save_prediction(result)
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            result = {
                'status': 'error',
                'message': str(e)
            }
        
        return result
    
    def save_prediction(self, result: Dict[str, Any]):
        """
        保存预测结果
        
        Parameters
        ----------
        result : dict
            预测结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result['model']}_{result['target_date']}_{timestamp}"
        
        # 保存JSON
        json_path = os.path.join(self.config.result_paths['predictions'], f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存CSV（便于查看）
        csv_path = os.path.join(self.config.result_paths['predictions'], f"{filename}.csv")
        df = pd.DataFrame({
            'Hour': result['predictions']['hours'],
            'Predicted_Price': result['predictions']['prices']
        })
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"预测结果已保存:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV: {csv_path}")
    
    def batch_predict(self, model_names: List[str], target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        批量预测（使用多个模型）
        
        Parameters
        ----------
        model_names : list
            模型名称列表
        target_date : str, optional
            目标预测日期
            
        Returns
        -------
        results : dict
            各模型的预测结果
        """
        results = {}
        
        logger.info(f"开始批量预测，使用 {len(model_names)} 个模型")
        
        for model_name in model_names:
            result = self.predict(model_name, target_date)
            results[model_name] = result
        
        # 计算集成预测（简单平均）
        successful_predictions = []
        for model_name, result in results.items():
            if result['status'] == 'success':
                successful_predictions.append(result['predictions']['prices'])
        
        if len(successful_predictions) > 0:
            ensemble_prices = np.mean(successful_predictions, axis=0)
            results['Ensemble'] = {
                'status': 'success',
                'model': 'Ensemble (Average)',
                'target_date': target_date or (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'predictions': {
                    'hours': list(range(24)),
                    'prices': ensemble_prices.tolist()
                },
                'statistics': {
                    'min_price': float(np.min(ensemble_prices)),
                    'max_price': float(np.max(ensemble_prices)),
                    'mean_price': float(np.mean(ensemble_prices)),
                    'std_price': float(np.std(ensemble_prices))
                }
            }
            
            logger.info(f"集成预测完成:")
            logger.info(f"  平均价格: {results['Ensemble']['statistics']['mean_price']:.2f}")
        
        return results
    
    def print_prediction(self, result: Dict[str, Any]):
        """
        打印预测结果
        
        Parameters
        ----------
        result : dict
            预测结果
        """
        if result['status'] != 'success':
            print(f"\n预测失败: {result.get('message', '未知错误')}")
            return
        
        print("\n" + "=" * 60)
        print(f"电价预测结果")
        print("=" * 60)
        print(f"模型: {result['model']}")
        print(f"目标日期: {result['target_date']}")
        print(f"预测时间: {result['prediction_time']}")
        print("-" * 60)
        print(f"{'小时':<10}{'预测价格':<15}")
        print("-" * 60)
        
        for hour, price in zip(result['predictions']['hours'], result['predictions']['prices']):
            print(f"{hour:02d}:00      {price:>10.2f}")
        
        print("-" * 60)
        print(f"最低价格: {result['statistics']['min_price']:.2f}")
        print(f"最高价格: {result['statistics']['max_price']:.2f}")
        print(f"平均价格: {result['statistics']['mean_price']:.2f}")
        print(f"价格标准差: {result['statistics']['std_price']:.2f}")
        print("=" * 60)


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='电价预测')
    parser.add_argument('--model', type=str, required=True,
                       help='模型名称')
    parser.add_argument('--date', type=str, default=None,
                       help='目标日期 (YYYY-MM-DD，默认明天)')
    parser.add_argument('--batch', nargs='+', default=None,
                       help='批量预测，指定多个模型')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 创建预测器
    predictor = Predictor(config)
    
    # 批量预测
    if args.batch:
        results = predictor.batch_predict(args.batch, args.date)
        for model_name, result in results.items():
            print(f"\n{'='*60}")
            print(f"模型: {model_name}")
            print(f"{'='*60}")
            predictor.print_prediction(result)
    else:
        # 单模型预测
        result = predictor.predict(args.model, args.date)
        predictor.print_prediction(result)


if __name__ == '__main__':
    main()
