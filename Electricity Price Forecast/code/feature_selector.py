"""
特征选择管理器
支持根据模型配置自动选择特征
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    特征选择器
    根据配置文件为不同模型选择特征
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化特征选择器
        
        Parameters
        ----------
        config_path : str, optional
            特征配置文件路径，默认使用同级目录下的feature_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'feature_config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        logger.info(f"特征选择器初始化完成，配置文件: {self.config_path}")
    
    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"特征配置加载成功")
            return config
        except Exception as e:
            logger.error(f"加载特征配置失败: {e}")
            raise
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        获取所有特征组定义
        
        Returns
        -------
        feature_groups : Dict[str, List[str]]
            特征组名称到特征列表的映射
        """
        return self.config.get('feature_groups', {})
    
    def get_model_features(self, model_name: str) -> Dict:
        """
        获取指定模型的特征配置
        
        Parameters
        ----------
        model_name : str
            模型名称
            
        Returns
        -------
        model_config : Dict
            模型特征配置，包含feature_groups等
        """
        model_features = self.config.get('model_features', {})
        
        if model_name in model_features:
            return model_features[model_name]
        else:
            logger.warning(f"模型 {model_name} 未在配置中定义，使用默认配置")
            return self.config.get('default', {'feature_groups': ['time_basic', 'market_basic']})
    
    def select_features_for_model(self, model_name: str, 
                                   available_features: List[str]) -> List[str]:
        """
        为指定模型选择特征
        
        Parameters
        ----------
        model_name : str
            模型名称
        available_features : List[str]
            可用的特征列表（从特征文件中读取的所有列）
            
        Returns
        -------
        selected_features : List[str]
            选中的特征列表
        """
        model_config = self.get_model_features(model_name)
        
        # 获取特征组
        feature_groups = self.get_feature_groups()
        group_names = model_config.get('feature_groups', [])
        
        # 从特征组收集特征
        selected = set()
        for group_name in group_names:
            if group_name in feature_groups:
                selected.update(feature_groups[group_name])
            else:
                logger.warning(f"特征组 {group_name} 不存在")
        
        # 添加额外包含的特征
        include_features = model_config.get('include_features', [])
        selected.update(include_features)
        
        # 移除排除的特征
        exclude_features = model_config.get('exclude_features', [])
        selected.difference_update(exclude_features)
        
        # 如果指定了完全自定义特征，则使用自定义特征
        custom_features = model_config.get('custom_features', [])
        if custom_features:
            selected = set(custom_features)
        
        # 过滤掉不存在的特征
        available_set = set(available_features)
        selected = selected.intersection(available_set)
        
        # 移除目标变量列（如果存在）
        target_patterns = ['Price_H', '平均出清价格']
        selected = [f for f in selected if not any(p in f for p in target_patterns)]
        
        selected = sorted(list(selected))
        
        logger.info(f"模型 {model_name} 选中 {len(selected)} 个特征")
        logger.info(f"  特征组: {group_names}")
        if len(selected) <= 10:
            logger.info(f"  特征列表: {selected}")
        else:
            logger.info(f"  特征列表: {selected[:5]}...{selected[-5:]} (共{len(selected)}个)")
        
        return selected
    
    def get_model_feature_info(self, model_name: str) -> Dict:
        """
        获取模型特征信息
        
        Parameters
        ----------
        model_name : str
            模型名称
            
        Returns
        -------
        info : Dict
            特征信息，包含描述、是否需要标准化等
        """
        model_config = self.get_model_features(model_name)
        
        return {
            'description': model_config.get('description', ''),
            'normalize': model_config.get('normalize', False),
            'sequence_length': model_config.get('sequence_length', None),
            'feature_groups': model_config.get('feature_groups', [])
        }
    
    def list_all_models(self) -> List[str]:
        """
        列出所有配置了特征的模型
        
        Returns
        -------
        models : List[str]
            模型名称列表
        """
        return list(self.config.get('model_features', {}).keys())
    
    def list_feature_groups(self) -> List[str]:
        """
        列出所有特征组
        
        Returns
        -------
        groups : List[str]
            特征组名称列表
        """
        return list(self.get_feature_groups().keys())
    
    def print_feature_summary(self, model_name: Optional[str] = None):
        """
        打印特征配置摘要
        
        Parameters
        ----------
        model_name : str, optional
            指定模型名称，不指定则打印所有模型
        """
        print("\n" + "=" * 60)
        print("特征配置摘要")
        print("=" * 60)
        
        # 打印特征组
        print("\n【特征组】")
        for group_name, features in self.get_feature_groups().items():
            print(f"  {group_name}: {len(features)} 个特征")
        
        # 打印模型配置
        print("\n【模型特征配置】")
        models = [model_name] if model_name else self.list_all_models()
        
        for name in models:
            config = self.get_model_features(name)
            print(f"\n  {name}:")
            print(f"    描述: {config.get('description', 'N/A')}")
            print(f"    特征组: {config.get('feature_groups', [])}")
            if config.get('normalize'):
                print(f"    需要标准化: 是")
            if config.get('sequence_length'):
                print(f"    序列长度: {config['sequence_length']}")
        
        print("\n" + "=" * 60)


def main():
    """测试特征选择器"""
    selector = FeatureSelector()
    
    # 打印配置摘要
    selector.print_feature_summary()
    
    # 模拟可用特征
    available_features = [
        'month', 'day_of_week', 'is_weekend', 'hour', 'is_peak_hour',
        'hour_sin', 'hour_cos', 'day_of_month', 'quarter',
        '系统负荷-日前', '系统负荷-实时', '风电出力-日前', '风电出力-实时',
        '光伏出力-日前', '光伏出力-实时', '水电出力-日前', '水电出力-实时',
        '平均出清价格-实时（元/MWh）_lag_1', '平均出清价格-实时（元/MWh）_lag_2',
        '平均出清价格-实时（元/MWh）_lag_7',
        '平均出清价格-实时（元/MWh）_rolling_mean_7d',
        'Price_H00', 'Price_H01', 'Price_H02'  # 目标变量
    ]
    
    print("\n" + "=" * 60)
    print("测试特征选择")
    print("=" * 60)
    
    # 测试不同模型的特征选择
    test_models = ['Lasso', 'XGBoost', 'LSTM']
    
    for model_name in test_models:
        print(f"\n【{model_name}】")
        features = selector.select_features_for_model(model_name, available_features)
        info = selector.get_model_feature_info(model_name)
        print(f"  选中 {len(features)} 个特征")
        print(f"  描述: {info['description']}")
        print(f"  需要标准化: {info['normalize']}")
        print(f"  特征: {features}")


if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
