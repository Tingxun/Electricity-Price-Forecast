"""
模型模块
包含各种电价预测模型的实现
"""

# 导入基础模型
from .base_model import BaseModel

# 导入线性模型
from .linear_models import (
    LinearRegressionModel,
    RidgeModel,
    LassoModel,
    ElasticNetModel,
    create_linear_model
)

# 导入树模型
from .tree_models import (
    RandomForestModel,
    GradientBoostingModel,
    create_tree_model
)

# 尝试导入XGBoost模型
try:
    from .tree_models import XGBoostModel
except ImportError:
    pass

# 神经网络模型 - 延迟导入以避免不必要的依赖加载
_neural_networks_loaded = False
MLPModel = None
LSTMModel = None
GRUModel = None
TransformerModel = None
create_neural_network = None

def _load_neural_networks():
    """延迟加载神经网络模型"""
    global _neural_networks_loaded, MLPModel, LSTMModel, GRUModel, TransformerModel, create_neural_network
    if not _neural_networks_loaded:
        from .neural_networks import (
            MLPModel as _MLPModel,
            LSTMModel as _LSTMModel,
            GRUModel as _GRUModel,
            TransformerModel as _TransformerModel,
            create_neural_network as _create_neural_network
        )
        MLPModel = _MLPModel
        LSTMModel = _LSTMModel
        GRUModel = _GRUModel
        TransformerModel = _TransformerModel
        create_neural_network = _create_neural_network
        _neural_networks_loaded = True

# 尝试导入epftoolbox模型
try:
    from .epftoolbox_wrapper import EPFModel, create_epf_model
except ImportError:
    pass

# 导出所有模型
__all__ = [
    'BaseModel',
    'LinearRegressionModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'create_linear_model',
    'create_tree_model',
]

# 如果XGBoost可用，添加到导出列表
try:
    __all__.append('XGBoostModel')
except NameError:
    pass

# 如果epftoolbox可用，添加到导出列表
try:
    __all__.extend(['EPFModel', 'create_epf_model'])
except NameError:
    pass
