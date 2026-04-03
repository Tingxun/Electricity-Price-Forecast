"""
基础模型类
定义所有预测模型的抽象基类，统一接口规范
"""

from abc import ABC, abstractmethod
import pickle
import os
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """
    电价预测模型基类
    
    所有具体的预测模型都需要继承此类并实现抽象方法
    """
    
    def __init__(self, name: str = "BaseModel", **kwargs):
        """
        初始化模型
        
        Parameters
        ----------
        name : str
            模型名称
        **kwargs : dict
            模型特定参数
        """
        self.name = name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X, y, **kwargs) -> 'BaseModel':
        """
        训练模型
        
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
        self : BaseModel
            返回训练好的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> Any:
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
        pass
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Parameters
        ----------
        path : str
            模型保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'name': self.name,
                'params': self.params,
                'is_fitted': self.is_fitted
            }, f)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str) -> 'BaseModel':
        """
        从文件加载模型
        
        Parameters
        ----------
        path : str
            模型文件路径
            
        Returns
        -------
        self : BaseModel
            返回加载后的模型实例
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.name = data['name']
            self.params = data['params']
            self.is_fitted = data['is_fitted']
        print(f"模型已从 {path} 加载")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns
        -------
        params : dict
            模型参数字典
        """
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        设置模型参数
        
        Parameters
        ----------
        **params : dict
            要设置的参数
            
        Returns
        -------
        self : BaseModel
            返回设置后的模型实例
        """
        self.params.update(params)
        return self
    
    def get_name(self) -> str:
        """
        获取模型名称
        
        Returns
        -------
        name : str
            模型名称
        """
        return self.name
