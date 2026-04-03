"""
epftoolbox包装器
包装epftoolbox库中的模型，使其符合BaseModel接口规范
"""

from .base_model import BaseModel
import numpy as np
from typing import Optional, Dict, Any

# 尝试导入epftoolbox，如果没有安装则跳过
try:
    from epftoolbox.models import LEAR
    has_epftoolbox = True
except ImportError:
    has_epftoolbox = False


class EPFModel(BaseModel):
    """
    epftoolbox模型包装器
    """
    
    def __init__(self, name: str = "EPFModel", model_type: str = "LEAR", **kwargs):
        """
        初始化epftoolbox模型
        
        Parameters
        ----------
        name : str
            模型名称
        model_type : str
            模型类型: 'LEAR' (线性自回归模型)
        **kwargs : dict
            模型参数
        """
        if not has_epftoolbox:
            raise ImportError("epftoolbox库未安装，请使用 'pip install epftoolbox' 安装")
        
        super().__init__(name=name, model_type=model_type, **kwargs)
        self.model_type = model_type
        
        # 创建epftoolbox模型
        self._create_model()
    
    def _create_model(self):
        """
        创建epftoolbox模型
        """
        model_map = {
            'LEAR': LEAR
        }
        
        if self.model_type not in model_map:
            raise ValueError(f"不支持的epftoolbox模型类型: {self.model_type}，支持的类型: {list(model_map.keys())}")
        
        # 移除model_type参数，避免传递给模型构造函数
        model_kwargs = self.params.copy()
        if 'model_type' in model_kwargs:
            del model_kwargs['model_type']
        
        # 创建模型
        self.model = model_map[self.model_type](**model_kwargs)
        print(f"epftoolbox {self.model_type} 模型已创建")
    
    def fit(self, X, y, **kwargs) -> 'EPFModel':
        """
        训练epftoolbox模型
        
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
        self : EPFModel
            返回训练好的模型实例
        """
        # 确保输入是numpy数组
        X = np.array(X)
        y = np.array(y)
        
        # LEAR模型使用recalibrate方法进行训练
        if self.model_type == 'LEAR':
            # LEAR需要特定的数据格式
            # 假设X和y已经是正确的格式
            self.model.recalibrate(X, y)
        else:
            # 其他模型使用通用的fit方法
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y, **kwargs)
            else:
                raise NotImplementedError(f"模型 {self.model_type} 没有实现训练方法")
        
        self.is_fitted = True
        print(f"epftoolbox {self.model_type} 模型训练完成")
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
        
        # 确保输入是numpy数组
        X = np.array(X)
        
        # 调用epftoolbox模型的predict方法
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Parameters
        ----------
        path : str
            模型保存路径
        """
        # 检查epftoolbox模型是否支持save方法
        if hasattr(self.model, 'save'):
            self.model.save(path)
            print(f"模型已保存到: {path}")
        else:
            # 如果不支持，使用默认的pickle保存
            super().save(path)
    
    def load(self, path: str) -> 'EPFModel':
        """
        从文件加载模型
        
        Parameters
        ----------
        path : str
            模型文件路径
            
        Returns
        -------
        self : EPFModel
            返回加载后的模型实例
        """
        # 检查epftoolbox模型是否支持load方法
        if hasattr(self.model, 'load'):
            self.model = self.model.__class__().load(path)
            self.is_fitted = True
            print(f"模型已从 {path} 加载")
        else:
            # 如果不支持，使用默认的pickle加载
            super().load(path)
        
        return self


# 模型工厂函数
def create_epf_model(model_type: str = "LEAR", **kwargs) -> BaseModel:
    """
    创建epftoolbox模型
    
    Parameters
    ----------
    model_type : str
        模型类型: 'LEAR' (线性自回归模型)
    **kwargs : dict
        模型参数
        
    Returns
    -------
    model : BaseModel
        创建的模型实例
    """
    if not has_epftoolbox:
        raise ImportError("epftoolbox库未安装，请使用 'pip install epftoolbox' 安装")
    
    return EPFModel(model_type=model_type, **kwargs)