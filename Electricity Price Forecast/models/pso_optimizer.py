"""
PSO (Particle Swarm Optimization) 优化算法实现
用于优化MLP填充器的特征选择和超参数
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Optional
import copy


class Particle:
    """粒子类，表示PSO中的一个粒子"""
    
    def __init__(self, n_features: int, n_hyperparams: int, 
                 feature_bounds: Tuple[int, int] = (0, 1),
                 hyperparam_bounds: List[Tuple[float, float]] = None):
        """
        初始化粒子
        
        Parameters
        ----------
        n_features : int
            特征数量（二进制编码长度）
        n_hyperparams : int
            超参数数量
        feature_bounds : Tuple[int, int]
            特征选择边界 (0, 1) 表示是否选择该特征
        hyperparam_bounds : List[Tuple[float, float]]
            每个超参数的边界范围
        """
        self.n_features = n_features
        self.n_hyperparams = n_hyperparams
        self.feature_bounds = feature_bounds
        self.hyperparam_bounds = hyperparam_bounds if hyperparam_bounds else []
        
        # 位置：前n_features位是特征选择（二进制），后面是超参数（连续值）
        self.position_features = np.random.randint(0, 2, n_features).astype(float)
        self.position_hyperparams = np.zeros(n_hyperparams)
        for i, (low, high) in enumerate(self.hyperparam_bounds):
            self.position_hyperparams[i] = np.random.uniform(low, high)
        
        # 速度
        self.velocity_features = np.random.randn(n_features) * 0.1
        self.velocity_hyperparams = np.random.randn(n_hyperparams) * 0.1
        
        # 最佳位置
        self.best_position_features = self.position_features.copy()
        self.best_position_hyperparams = self.position_hyperparams.copy()
        self.best_fitness = float('inf')
        
        # 当前适应度
        self.current_fitness = float('inf')
    
    def update_velocity(self, global_best_features: np.ndarray, 
                       global_best_hyperparams: np.ndarray,
                       w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
        """
        更新粒子速度
        
        Parameters
        ----------
        global_best_features : np.ndarray
            全局最佳特征选择
        global_best_hyperparams : np.ndarray
            全局最佳超参数
        w : float
            惯性权重
        c1 : float
            个体学习因子
        c2 : float
            社会学习因子
        """
        r1, r2 = np.random.random(2)
        
        # 更新特征选择速度（二进制部分使用特殊处理）
        cognitive_features = c1 * r1 * (self.best_position_features - self.position_features)
        social_features = c2 * r2 * (global_best_features - self.position_features)
        self.velocity_features = w * self.velocity_features + cognitive_features + social_features
        
        # 限制特征速度
        self.velocity_features = np.clip(self.velocity_features, -3, 3)
        
        # 更新超参数速度
        r1, r2 = np.random.random(2)
        cognitive_hyperparams = c1 * r1 * (self.best_position_hyperparams - self.position_hyperparams)
        social_hyperparams = c2 * r2 * (global_best_hyperparams - self.position_hyperparams)
        self.velocity_hyperparams = w * self.velocity_hyperparams + cognitive_hyperparams + social_hyperparams
    
    def update_position(self):
        """更新粒子位置"""
        # 使用sigmoid函数将速度映射到概率，进行二进制更新
        sigmoid = 1 / (1 + np.exp(-self.velocity_features))
        random_vals = np.random.random(self.n_features)
        self.position_features = (random_vals < sigmoid).astype(float)
        
        # 确保至少选择一个特征
        if np.sum(self.position_features) == 0:
            self.position_features[np.random.randint(0, self.n_features)] = 1
        
        # 更新超参数位置
        self.position_hyperparams += self.velocity_hyperparams
        
        # 限制超参数在边界内
        for i, (low, high) in enumerate(self.hyperparam_bounds):
            self.position_hyperparams[i] = np.clip(self.position_hyperparams[i], low, high)
    
    def evaluate(self, fitness_func: Callable) -> float:
        """
        评估粒子适应度
        
        Parameters
        ----------
        fitness_func : Callable
            适应度函数，接收特征选择和超参数，返回适应度值（越小越好）
        
        Returns
        -------
        fitness : float
            适应度值
        """
        self.current_fitness = fitness_func(
            self.position_features, 
            self.position_hyperparams
        )
        
        # 更新个体最佳
        if self.current_fitness < self.best_fitness:
            self.best_fitness = self.current_fitness
            self.best_position_features = self.position_features.copy()
            self.best_position_hyperparams = self.position_hyperparams.copy()
        
        return self.current_fitness
    
    def get_selected_features(self) -> List[int]:
        """获取被选中的特征索引"""
        return [i for i, val in enumerate(self.position_features) if val > 0.5]
    
    def get_hyperparams(self) -> Dict:
        """获取超参数字典"""
        return {
            'hidden_dim1': int(self.position_hyperparams[0]),
            'hidden_dim2': int(self.position_hyperparams[1]) if len(self.position_hyperparams) > 1 else 64,
            'lr': self.position_hyperparams[2] if len(self.position_hyperparams) > 2 else 0.001,
            'batch_size': int(self.position_hyperparams[3]) if len(self.position_hyperparams) > 3 else 64,
            'window_size': int(self.position_hyperparams[4]) if len(self.position_hyperparams) > 4 else 48,
        }


class PSOOptimizer:
    """PSO优化器类"""
    
    def __init__(self, n_features: int, n_hyperparams: int,
                 n_particles: int = 20, max_iter: int = 50,
                 hyperparam_bounds: List[Tuple[float, float]] = None,
                 w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
                 verbose: bool = True):
        """
        初始化PSO优化器
        
        Parameters
        ----------
        n_features : int
            特征数量
        n_hyperparams : int
            超参数数量
        n_particles : int
            粒子数量
        max_iter : int
            最大迭代次数
        hyperparam_bounds : List[Tuple[float, float]]
            超参数边界范围
        w : float
            惯性权重
        c1 : float
            个体学习因子
        c2 : float
            社会学习因子
        verbose : bool
            是否打印进度
        """
        self.n_features = n_features
        self.n_hyperparams = n_hyperparams
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.hyperparam_bounds = hyperparam_bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        # 初始化粒子群
        self.particles = []
        for _ in range(n_particles):
            particle = Particle(n_features, n_hyperparams, 
                              hyperparam_bounds=hyperparam_bounds)
            self.particles.append(particle)
        
        # 全局最佳
        self.global_best_features = None
        self.global_best_hyperparams = None
        self.global_best_fitness = float('inf')
        
        # 历史记录
        self.fitness_history = []
        self.best_fitness_history = []
    
    def optimize(self, fitness_func: Callable) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        执行PSO优化
        
        Parameters
        ----------
        fitness_func : Callable
            适应度函数，接收特征选择和超参数，返回适应度值
        
        Returns
        -------
        best_features : np.ndarray
            最佳特征选择
        best_hyperparams : np.ndarray
            最佳超参数
        best_fitness : float
            最佳适应度值
        """
        for iteration in range(self.max_iter):
            iter_fitness = []
            
            for i, particle in enumerate(self.particles):
                fitness = particle.evaluate(fitness_func)
                iter_fitness.append(fitness)
                
                # 更新全局最佳
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_features = particle.position_features.copy()
                    self.global_best_hyperparams = particle.position_hyperparams.copy()
                    
                    if self.verbose:
                        selected = particle.get_selected_features()
                        hyperparams = particle.get_hyperparams()
                        print(f"  迭代 {iteration+1}, 粒子 {i+1}: 新最佳适应度 = {fitness:.6f}")
                        print(f"    选中特征: {selected}, 超参数: {hyperparams}")
            
            # 记录历史
            self.fitness_history.append(iter_fitness)
            self.best_fitness_history.append(self.global_best_fitness)
            
            if self.verbose and (iteration + 1) % 5 == 0:
                avg_fitness = np.mean(iter_fitness)
                print(f"迭代 {iteration+1}/{self.max_iter}: 平均适应度 = {avg_fitness:.6f}, 最佳 = {self.global_best_fitness:.6f}")
            
            # 更新速度和位置
            for particle in self.particles:
                particle.update_velocity(
                    self.global_best_features,
                    self.global_best_hyperparams,
                    self.w, self.c1, self.c2
                )
                particle.update_position()
        
        if self.verbose:
            print(f"\nPSO优化完成！最佳适应度: {self.global_best_fitness:.6f}")
            print(f"最佳特征选择: {[i for i, v in enumerate(self.global_best_features) if v > 0.5]}")
        
        return self.global_best_features, self.global_best_hyperparams, self.global_best_fitness
    
    def get_best_solution(self) -> Dict:
        """获取最佳解决方案"""
        if self.global_best_features is None:
            return None
        
        selected_features = [i for i, v in enumerate(self.global_best_features) if v > 0.5]
        
        hyperparams = {}
        if len(self.global_best_hyperparams) >= 1:
            hyperparams['hidden_dim1'] = int(self.global_best_hyperparams[0])
        if len(self.global_best_hyperparams) >= 2:
            hyperparams['hidden_dim2'] = int(self.global_best_hyperparams[1])
        if len(self.global_best_hyperparams) >= 3:
            hyperparams['lr'] = self.global_best_hyperparams[2]
        if len(self.global_best_hyperparams) >= 4:
            hyperparams['batch_size'] = int(self.global_best_hyperparams[3])
        if len(self.global_best_hyperparams) >= 5:
            hyperparams['window_size'] = int(self.global_best_hyperparams[4])
        
        return {
            'features': selected_features,
            'hyperparams': hyperparams,
            'fitness': self.global_best_fitness
        }


def create_mlp_fitness_evaluator(X_train_list, y_train_list, X_val_list, y_val_list,
                                 feature_names: List[str], device: str = 'cuda'):
    """
    创建MLP适应度评估函数
    
    Parameters
    ----------
    X_train_list : List[np.ndarray]
        训练特征列表（3折交叉验证）
    y_train_list : List[np.ndarray]
        训练目标列表
    X_val_list : List[np.ndarray]
        验证特征列表
    y_val_list : List[np.ndarray]
        验证目标列表
    feature_names : List[str]
        特征名称列表
    device : str
        计算设备
    
    Returns
    -------
    fitness_func : Callable
        适应度评估函数
    """
    import sys
    import os
    from sklearn.metrics import mean_squared_error
    
    def fitness_func(feature_selection: np.ndarray, hyperparams: np.ndarray) -> float:
        """
        评估适应度（3折交叉验证的平均RMSE）
        
        Parameters
        ----------
        feature_selection : np.ndarray
            特征选择（二进制数组）
        hyperparams : np.ndarray
            超参数数组
        
        Returns
        -------
        fitness : float
            适应度值（平均RMSE，越小越好）
        """
        # 获取选中的特征索引
        selected_indices = [i for i, v in enumerate(feature_selection) if v > 0.5]
        
        if len(selected_indices) == 0:
            return float('inf')
        
        # 解析超参数
        hidden_dim1 = int(hyperparams[0]) if len(hyperparams) > 0 else 128
        hidden_dim2 = int(hyperparams[1]) if len(hyperparams) > 1 else 64
        lr = hyperparams[2] if len(hyperparams) > 2 else 0.001
        batch_size = int(hyperparams[3]) if len(hyperparams) > 3 else 64
        window_size = int(hyperparams[4]) if len(hyperparams) > 4 else 48
        
        # 确保超参数在合理范围内
        hidden_dim1 = max(16, min(512, hidden_dim1))
        hidden_dim2 = max(16, min(256, hidden_dim2))
        lr = max(1e-5, min(0.1, lr))
        batch_size = max(16, min(256, batch_size))
        window_size = max(12, min(96, window_size))
        
        # 3折交叉验证
        cv_rmse_list = []
        
        for fold in range(3):
            try:
                X_train_fold = X_train_list[fold][:, selected_indices]
                y_train_fold = y_train_list[fold]
                X_val_fold = X_val_list[fold][:, selected_indices]
                y_val_fold = y_val_list[fold]
                
                # 检查数据有效性
                if len(X_train_fold) < window_size + 1 or len(X_val_fold) == 0:
                    continue
                
                # 构建序列数据
                X_seq, y_seq = [], []
                for i in range(len(X_train_fold) - window_size):
                    X_seq.append(X_train_fold[i:i+window_size].flatten())
                    y_seq.append(y_train_fold[i+window_size])
                
                if len(X_seq) < 10:
                    continue
                
                X_seq = np.array(X_seq, dtype=np.float32)
                y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
                
                # 导入MLP模型
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                from models.neural_networks import MLPModel
                
                # 创建并训练模型
                model = MLPModel(
                    input_dim=X_seq.shape[1],
                    output_dim=1,
                    hidden_dims=[hidden_dim1, hidden_dim2],
                    batch_size=batch_size,
                    epochs=100,
                    lr=lr,
                    device=device,
                    verbose=False
                )
                
                model.fit(X_seq, y_seq)
                
                # 验证
                X_val_seq, y_val_seq = [], []
                for i in range(len(X_val_fold) - window_size):
                    X_val_seq.append(X_val_fold[i:i+window_size].flatten())
                    y_val_seq.append(y_val_fold[i+window_size])
                
                if len(X_val_seq) == 0:
                    continue
                
                X_val_seq = np.array(X_val_seq, dtype=np.float32)
                y_val_seq = np.array(y_val_seq, dtype=np.float32)
                
                predictions = model.predict(X_val_seq).flatten()
                rmse = np.sqrt(mean_squared_error(y_val_seq, predictions))
                cv_rmse_list.append(rmse)
                
            except Exception as e:
                # 训练失败返回一个较大的值
                continue
        
        if len(cv_rmse_list) == 0:
            return float('inf')
        
        # 返回平均RMSE作为适应度（越小越好）
        avg_rmse = np.mean(cv_rmse_list)
        
        # 添加特征选择惩罚（避免选择过多特征）
        n_selected = len(selected_indices)
        penalty = 0.001 * n_selected  # 轻微惩罚
        
        return avg_rmse + penalty
    
    return fitness_func
