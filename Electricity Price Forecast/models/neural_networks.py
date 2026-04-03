"""
神经网络模型实现
包含多种深度学习模型，支持多输出回归（24点预测）
"""

from .base_model import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, Tuple
import time


class PyTorchModel(BaseModel, nn.Module):
    """
    PyTorch模型基类
    """
    
    def __init__(self, name: str = "PyTorchModel", **kwargs):
        """
        初始化PyTorch模型
        
        Parameters
        ----------
        name : str
            模型名称
        **kwargs : dict
            模型参数
        """
        BaseModel.__init__(self, name=name, **kwargs)
        nn.Module.__init__(self)
        
        # 默认参数
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        self.lr = kwargs.get('lr', 0.001)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
    
    def parameters(self, recurse: bool = True):
        """
        获取模型参数
        """
        if hasattr(self, 'model') and self.model is not self:
            return self.model.parameters(recurse=recurse)
        else:
            return super().parameters(recurse=recurse)
    
    def fit(self, X, y, **kwargs) -> 'PyTorchModel':
        """
        训练PyTorch模型
        
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
        self : PyTorchModel
            返回训练好的模型实例
        """
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        if self.model is self:
            # 如果模型是自身（LSTM, GRU, Transformer）
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            # 如果模型是nn.Sequential（MLP）
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 训练循环
        start_time = time.time()
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                avg_loss = running_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        training_time = time.time() - start_time
        print(f"{self.name} 训练完成，耗时: {training_time:.2f} 秒")
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
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            if self.model is self:
                # 如果模型是自身（LSTM, GRU, Transformer）
                outputs = self(X_tensor)
            else:
                # 如果模型是nn.Sequential（MLP）
                outputs = self.model(X_tensor)
        
        # 转换回numpy数组
        return outputs.cpu().numpy()
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Parameters
        ----------
        path : str
            模型保存路径
        """
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态
        if self.model is self:
            # 如果模型是自身（LSTM, GRU, Transformer）
            torch.save({
                'model_state_dict': self.state_dict(),
                'name': self.name,
                'params': self.params,
                'is_fitted': self.is_fitted
            }, path)
        else:
            # 如果模型是nn.Sequential（MLP）
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'name': self.name,
                'params': self.params,
                'is_fitted': self.is_fitted
            }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str) -> 'PyTorchModel':
        """
        从文件加载模型
        
        Parameters
        ----------
        path : str
            模型文件路径
            
        Returns
        -------
        self : PyTorchModel
            返回加载后的模型实例
        """
        import torch
        
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is self:
            # 如果模型是自身（LSTM, GRU, Transformer）
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果模型是nn.Sequential（MLP）
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.name = checkpoint['name']
        self.params = checkpoint['params']
        self.is_fitted = checkpoint['is_fitted']
        print(f"模型已从 {path} 加载")
        return self


class MLPModel(PyTorchModel):
    """
    多层感知器模型
    """
    
    def __init__(self, name: str = "MLP", input_dim: int = 100, output_dim: int = 24, 
                 hidden_dims: list = None, **kwargs):
        """
        初始化多层感知器模型
        
        Parameters
        ----------
        name : str
            模型名称
        input_dim : int
            输入特征维度
        output_dim : int
            输出维度（24点预测）
        hidden_dims : list
            隐藏层维度列表
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 默认隐藏层
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        # 构建模型
        layers = []
        current_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # 添加输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        print(f"MLP模型已创建: {input_dim} -> {hidden_dims} -> {output_dim}")


class LSTMModel(PyTorchModel):
    """
    LSTM模型
    """
    
    def __init__(self, name: str = "LSTM", input_dim: int = 100, output_dim: int = 24, 
                 hidden_dim: int = 64, num_layers: int = 2, **kwargs):
        """
        初始化LSTM模型
        
        Parameters
        ----------
        name : str
            模型名称
        input_dim : int
            输入特征维度
        output_dim : int
            输出维度（24点预测）
        hidden_dim : int
            隐藏层维度
        num_layers : int
            LSTM层数
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 构建LSTM模型组件
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)
        
        # 将模型设置为自身，因为我们需要自定义forward方法
        self.model = self
        
        print(f"LSTM模型已创建: {input_dim} -> {hidden_dim} (x{num_layers}) -> {output_dim}")
    
    def forward(self, x):
        """
        前向传播
        """
        # LSTM的输出是 (output, (h_n, c_n))
        output, _ = self.lstm(x)
        # 取最后一个时间步的输出
        output = output[:, -1, :]
        # 全连接层输出
        output = self.fc(output)
        return output
    
    def fit(self, X, y, **kwargs) -> 'LSTMModel':
        """
        训练LSTM模型
        
        Parameters
        ----------
        X : array-like
            训练特征，形状为 (batch, seq_len, features)
        y : array-like
            训练目标
        **kwargs : dict
            额外的训练参数
            
        Returns
        -------
        self : LSTMModel
            返回训练好的模型实例
        """
        # 确保输入是3D张量 (batch, seq_len, features)
        if len(X.shape) == 2:
            # 如果输入是2D，添加时间维度
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return super().fit(X, y, **kwargs)
    
    def predict(self, X) -> np.ndarray:
        """
        使用LSTM模型进行预测
        
        Parameters
        ----------
        X : array-like
            预测特征
            
        Returns
        -------
        predictions : array-like
            预测结果
        """
        # 确保输入是3D张量 (batch, seq_len, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return super().predict(X)


class GRUModel(PyTorchModel):
    """
    GRU模型
    """
    
    def __init__(self, name: str = "GRU", input_dim: int = 100, output_dim: int = 24, 
                 hidden_dim: int = 64, num_layers: int = 2, **kwargs):
        """
        初始化GRU模型
        
        Parameters
        ----------
        name : str
            模型名称
        input_dim : int
            输入特征维度
        output_dim : int
            输出维度（24点预测）
        hidden_dim : int
            隐藏层维度
        num_layers : int
            GRU层数
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 构建GRU模型组件
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)
        
        # 将模型设置为自身，因为我们需要自定义forward方法
        self.model = self
        
        print(f"GRU模型已创建: {input_dim} -> {hidden_dim} (x{num_layers}) -> {output_dim}")
    
    def forward(self, x):
        """
        前向传播
        """
        # GRU的输出是 (output, h_n)
        output, _ = self.gru(x)
        # 取最后一个时间步的输出
        output = output[:, -1, :]
        # 全连接层输出
        output = self.fc(output)
        return output
    
    def fit(self, X, y, **kwargs) -> 'GRUModel':
        """
        训练GRU模型
        
        Parameters
        ----------
        X : array-like
            训练特征，形状为 (batch, seq_len, features)
        y : array-like
            训练目标
        **kwargs : dict
            额外的训练参数
            
        Returns
        -------
        self : GRUModel
            返回训练好的模型实例
        """
        # 确保输入是3D张量 (batch, seq_len, features)
        if len(X.shape) == 2:
            # 如果输入是2D，添加时间维度
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return super().fit(X, y, **kwargs)
    
    def predict(self, X) -> np.ndarray:
        """
        使用GRU模型进行预测
        
        Parameters
        ----------
        X : array-like
            预测特征
            
        Returns
        -------
        predictions : array-like
            预测结果
        """
        # 确保输入是3D张量 (batch, seq_len, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return super().predict(X)


class TransformerModel(PyTorchModel):
    """
    Transformer模型
    """
    
    def __init__(self, name: str = "Transformer", input_dim: int = 100, output_dim: int = 24, 
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, **kwargs):
        """
        初始化Transformer模型
        
        Parameters
        ----------
        name : str
            模型名称
        input_dim : int
            输入特征维度
        output_dim : int
            输出维度（24点预测）
        d_model : int
            Transformer模型维度
        nhead : int
            多头注意力头数
        num_layers : int
            Transformer层数
        dim_feedforward : int
            前馈网络维度
        **kwargs : dict
            模型参数
        """
        super().__init__(name=name, **kwargs)
        
        # 构建Transformer模型组件
        self.embedding = nn.Linear(input_dim, d_model).to(self.device)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        ).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        ).to(self.device)
        self.fc = nn.Linear(d_model, output_dim).to(self.device)
        
        # 将模型设置为自身，因为我们需要自定义forward方法
        self.model = self
        
        print(f"Transformer模型已创建: {input_dim} -> {d_model} (nhead={nhead}, layers={num_layers}) -> {output_dim}")
    
    def forward(self, x):
        """
        前向传播
        """
        # 输入嵌入
        x = self.embedding(x)
        # Transformer编码
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        # 全连接层输出
        x = self.fc(x)
        return x
    
    def fit(self, X, y, **kwargs) -> 'TransformerModel':
        """
        训练Transformer模型
        
        Parameters
        ----------
        X : array-like
            训练特征，形状为 (batch, seq_len, features)
        y : array-like
            训练目标
        **kwargs : dict
            额外的训练参数
            
        Returns
        -------
        self : TransformerModel
            返回训练好的模型实例
        """
        # 确保输入是3D张量 (batch, seq_len, features)
        if len(X.shape) == 2:
            # 如果输入是2D，添加时间维度
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return super().fit(X, y, **kwargs)
    
    def predict(self, X) -> np.ndarray:
        """
        使用Transformer模型进行预测
        
        Parameters
        ----------
        X : array-like
            预测特征
            
        Returns
        -------
        predictions : array-like
            预测结果
        """
        # 确保输入是3D张量 (batch, seq_len, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return super().predict(X)


# 模型工厂函数
def create_neural_network(model_type: str, **kwargs) -> BaseModel:
    """
    创建神经网络模型
    
    Parameters
    ----------
    model_type : str
        模型类型: 'mlp', 'lstm', 'gru', 'transformer'
    **kwargs : dict
        模型参数
        
    Returns
    -------
    model : BaseModel
        创建的模型实例
    """
    model_map = {
        'mlp': MLPModel,
        'lstm': LSTMModel,
        'gru': GRUModel,
        'transformer': TransformerModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model_map[model_type](**kwargs)