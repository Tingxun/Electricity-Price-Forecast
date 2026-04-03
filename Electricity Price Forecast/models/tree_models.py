"""
树模型实现
包含多种树模型，支持多输出回归（24点预测）
"""

from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from typing import Optional, Dict, Any

# 尝试导入XGBoost，如果没有安装则跳过
try:
    from xgboost import XGBRegressor
    has_xgboost = True
except ImportError:
    has_xgboost = False


class RandomForestModel(BaseModel):
    """
    随机森林回归模型
    """
    
    def __init__(self, name: str = "RandomForest", multi_output: bool = True, n_estimators: int = 100, **kwargs):
        """
        初始化随机森林回归模型
        
        Parameters
        ----------
        name : str
            模型名称
        multi_output : bool
            是否支持多输出（24点预测）
        n_estimators : int
            树的数量
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, n_estimators=n_estimators, **kwargs)
        self.multi_output = multi_output
        
        # 创建基础模型
        base_model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
        
        # 如果需要多输出，使用MultiOutputRegressor包装
        if multi_output:
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model
    
    def fit(self, X, y, **kwargs) -> 'RandomForestModel':
        """
        训练随机森林回归模型
        
        Parameters
        ----------
        X : array-like
            训练特征
        y : array-like
            训练目标
        **kwargs : dict
            额外的训练参数
            
        Returns
        -------
        self : RandomForestModel
            返回训练好的模型实例
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        print(f"随机森林模型 {self.name} 训练完成")
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        使用模型进行预测
        
        Parameters
        ----------
        X : array-like
            预测特征
            
        Returns
        -------
        predictions : array-like
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns
        -------
        feature_importance : array-like
            特征重要性
        """
        if not self.is_fitted:
            return None
        
        if self.multi_output:
            # 对于多输出模型，返回每个输出的特征重要性的平均值
            importances = np.array([estimator.feature_importances_ for estimator in self.model.estimators_])
            return np.mean(importances, axis=0)
        else:
            return self.model.feature_importances_


class GradientBoostingModel(BaseModel):
    """
    梯度提升树回归模型
    """
    
    def __init__(self, name: str = "GradientBoosting", multi_output: bool = True, n_estimators: int = 100, **kwargs):
        """
        初始化梯度提升树回归模型
        
        Parameters
        ----------
        name : str
            模型名称
        multi_output : bool
            是否支持多输出（24点预测）
        n_estimators : int
            树的数量
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, n_estimators=n_estimators, **kwargs)
        self.multi_output = multi_output
        
        # 创建基础模型
        base_model = GradientBoostingRegressor(n_estimators=n_estimators, **kwargs)
        
        # 如果需要多输出，使用MultiOutputRegressor包装
        if multi_output:
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model
    
    def fit(self, X, y, **kwargs) -> 'GradientBoostingModel':
        """
        训练梯度提升树回归模型
        
        Parameters
        ----------
        X : array-like
            训练特征
        y : array-like
            训练目标
        **kwargs : dict
            额外的训练参数
            
        Returns
        -------
        self : GradientBoostingModel
            返回训练好的模型实例
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        print(f"梯度提升树模型 {self.name} 训练完成")
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        使用模型进行预测
        
        Parameters
        ----------
        X : array-like
            预测特征
            
        Returns
        -------
        predictions : array-like
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns
        -------
        feature_importance : array-like
            特征重要性
        """
        if not self.is_fitted:
            return None
        
        if self.multi_output:
            # 对于多输出模型，返回每个输出的特征重要性的平均值
            importances = np.array([estimator.feature_importances_ for estimator in self.model.estimators_])
            return np.mean(importances, axis=0)
        else:
            return self.model.feature_importances_


if has_xgboost:
    class XGBoostModel(BaseModel):
        """
        XGBoost回归模型
        """
        
        def __init__(self, name: str = "XGBoost", multi_output: bool = True, n_estimators: int = 100, **kwargs):
            """
            初始化XGBoost回归模型
            
            Parameters
            ----------
            name : str
                模型名称
            multi_output : bool
                是否支持多输出（24点预测）
            n_estimators : int
                树的数量
            **kwargs : dict
                模型参数
            """
            super().__init__(name=name, n_estimators=n_estimators, **kwargs)
            self.multi_output = multi_output
            
            # 创建基础模型
            base_model = XGBRegressor(n_estimators=n_estimators, **kwargs)
            
            # 如果需要多输出，使用MultiOutputRegressor包装
            if multi_output:
                self.model = MultiOutputRegressor(base_model)
            else:
                self.model = base_model
        
        def fit(self, X, y, **kwargs) -> 'XGBoostModel':
            """
            训练XGBoost回归模型
            
            Parameters
            ----------
            X : array-like
                训练特征
            y : array-like
                训练目标
            **kwargs : dict
                额外的训练参数
                
            Returns
            -------
            self : XGBoostModel
                返回训练好的模型实例
            """
            self.model.fit(X, y, **kwargs)
            self.is_fitted = True
            print(f"XGBoost模型 {self.name} 训练完成")
            return self
        
        def predict(self, X) -> np.ndarray:
            """
            使用模型进行预测
            
            Parameters
            ----------
            X : array-like
                预测特征
                
            Returns
            -------
            predictions : array-like
                预测结果
            """
            if not self.is_fitted:
                raise ValueError("模型尚未训练，请先调用fit方法")
            
            return self.model.predict(X)
        
        def get_feature_importance(self) -> Optional[np.ndarray]:
            """
            获取特征重要性
            
            Returns
            -------
            feature_importance : array-like
                特征重要性
            """
            if not self.is_fitted:
                return None
            
            if self.multi_output:
                # 对于多输出模型，返回每个输出的特征重要性的平均值
                importances = np.array([estimator.feature_importances_ for estimator in self.model.estimators_])
                return np.mean(importances, axis=0)
            else:
                return self.model.feature_importances_


# 模型工厂函数
def create_tree_model(model_type: str, **kwargs) -> BaseModel:
    """
    创建树模型
    
    Parameters
    ----------
    model_type : str
        模型类型: 'random_forest', 'gradient_boosting', 'xgboost'
    **kwargs : dict
        模型参数
        
    Returns
    -------
    model : BaseModel
        创建的模型实例
    """
    model_map = {
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
    }
    
    # 如果XGBoost可用，添加到模型映射
    if has_xgboost:
        model_map['xgboost'] = XGBoostModel
    
    if model_type not in model_map:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model_map[model_type](**kwargs)
