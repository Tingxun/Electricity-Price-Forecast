"""
线性模型实现
包含多种线性回归模型，支持多输出回归（24点预测）
"""

from .base_model import BaseModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from typing import Optional, Dict, Any


class LinearRegressionModel(BaseModel):
    """
    线性回归模型
    """
    
    def __init__(self, name: str = "LinearRegression", multi_output: bool = True, **kwargs):
        """
        初始化线性回归模型
        
        Parameters
        ----------
        name : str
            模型名称
        multi_output : bool
            是否支持多输出（24点预测）
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, **kwargs)
        self.multi_output = multi_output
        
        # 创建基础模型
        base_model = LinearRegression(**kwargs)
        
        # 如果需要多输出，使用MultiOutputRegressor包装
        if multi_output:
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model
    
    def fit(self, X, y, **kwargs) -> 'LinearRegressionModel':
        """
        训练线性回归模型
        
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
        self : LinearRegressionModel
            返回训练好的模型实例
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        print(f"线性回归模型 {self.name} 训练完成")
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
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        获取模型系数
        
        Returns
        -------
        coefficients : array-like
            模型系数
        """
        if not self.is_fitted:
            return None
        
        if self.multi_output:
            # 对于多输出模型，返回每个输出的系数
            return np.array([estimator.coef_ for estimator in self.model.estimators_])
        else:
            return self.model.coef_


class RidgeModel(BaseModel):
    """
    岭回归模型
    """
    
    def __init__(self, name: str = "Ridge", multi_output: bool = True, alpha: float = 1.0, **kwargs):
        """
        初始化岭回归模型
        
        Parameters
        ----------
        name : str
            模型名称
        multi_output : bool
            是否支持多输出（24点预测）
        alpha : float
            正则化强度
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, alpha=alpha, **kwargs)
        self.multi_output = multi_output
        
        # 创建基础模型
        base_model = Ridge(alpha=alpha, **kwargs)
        
        # 如果需要多输出，使用MultiOutputRegressor包装
        if multi_output:
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model
    
    def fit(self, X, y, **kwargs) -> 'RidgeModel':
        """
        训练岭回归模型
        
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
        self : RidgeModel
            返回训练好的模型实例
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        print(f"岭回归模型 {self.name} 训练完成")
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


class LassoModel(BaseModel):
    """
    Lasso回归模型
    """
    
    def __init__(self, name: str = "Lasso", multi_output: bool = True, alpha: float = 1.0, **kwargs):
        """
        初始化Lasso回归模型
        
        Parameters
        ----------
        name : str
            模型名称
        multi_output : bool
            是否支持多输出（24点预测）
        alpha : float
            正则化强度
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, alpha=alpha, **kwargs)
        self.multi_output = multi_output
        
        # 创建基础模型
        base_model = Lasso(alpha=alpha, **kwargs)
        
        # 如果需要多输出，使用MultiOutputRegressor包装
        if multi_output:
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model
    
    def fit(self, X, y, **kwargs) -> 'LassoModel':
        """
        训练Lasso回归模型
        
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
        self : LassoModel
            返回训练好的模型实例
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        print(f"Lasso回归模型 {self.name} 训练完成")
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


class ElasticNetModel(BaseModel):
    """
    Elastic Net回归模型
    """
    
    def __init__(self, name: str = "ElasticNet", multi_output: bool = True, 
                 alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs):
        """
        初始化Elastic Net回归模型
        
        Parameters
        ----------
        name : str
            模型名称
        multi_output : bool
            是否支持多输出（24点预测）
        alpha : float
            正则化强度
        l1_ratio : float
            L1正则化比例
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        self.multi_output = multi_output
        
        # 创建基础模型
        base_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        
        # 如果需要多输出，使用MultiOutputRegressor包装
        if multi_output:
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model
    
    def fit(self, X, y, **kwargs) -> 'ElasticNetModel':
        """
        训练Elastic Net回归模型
        
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
        self : ElasticNetModel
            返回训练好的模型实例
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        print(f"Elastic Net回归模型 {self.name} 训练完成")
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


# 模型工厂函数
def create_linear_model(model_type: str, **kwargs) -> BaseModel:
    """
    创建线性模型
    
    Parameters
    ----------
    model_type : str
        模型类型: 'linear', 'ridge', 'lasso', 'elastic_net'
    **kwargs : dict
        模型参数
        
    Returns
    -------
    model : BaseModel
        创建的模型实例
    """
    model_map = {
        'linear': LinearRegressionModel,
        'ridge': RidgeModel,
        'lasso': LassoModel,
        'elastic_net': ElasticNetModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model_map[model_type](**kwargs)