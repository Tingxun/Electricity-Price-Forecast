"""
PSO (Particle Swarm Optimization) 优化算法实现
使用pyswarms库进行特征选择和超参数优化
"""

import numpy as np
import pyswarms as ps
from typing import List, Dict, Tuple, Callable


class PSOOptimizer:
    """PSO优化器类 - 使用pyswarm库"""

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
        self.hyperparam_bounds = hyperparam_bounds if hyperparam_bounds else []
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose

        # 全局最佳
        self.global_best_features = None
        self.global_best_hyperparams = None
        self.global_best_fitness = float('inf')

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
        # 构建边界
        lb = np.array([0.0] * self.n_features + [b[0] for b in self.hyperparam_bounds])
        ub = np.array([1.0] * self.n_features + [b[1] for b in self.hyperparam_bounds])

        # pyswarms的适应度函数需要接收整个粒子群 (n_particles, n_dimensions)
        def pso_fitness_swarm(X):
            """计算整个粒子群的适应度"""
            n_particles = X.shape[0]
            fitness_values = np.zeros(n_particles)

            for i in range(n_particles):
                x = X[i]
                feature_selection = (x[:self.n_features] > 0.5).astype(float)
                hyperparams = x[self.n_features:]

                # 确保至少选择一个特征
                if np.sum(feature_selection) == 0:
                    feature_selection[np.random.randint(0, self.n_features)] = 1

                fitness_values[i] = fitness_func(feature_selection, hyperparams)

            return fitness_values

        if self.verbose:
            print(f"\n  使用pyswarms进行PSO优化 (粒子数: {self.n_particles}, 迭代: {self.max_iter})...")

        # 创建pyswarms优化器
        options = {
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w
        }

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(lb),
            options=options,
            bounds=(lb, ub)
        )

        # 执行优化
        fopt, xopt = optimizer.optimize(
            pso_fitness_swarm,
            iters=self.max_iter,
            verbose=self.verbose
        )

        # 分离最佳特征选择和超参数
        self.global_best_features = (xopt[:self.n_features] > 0.5).astype(float)
        self.global_best_hyperparams = xopt[self.n_features:]
        self.global_best_fitness = fopt

        if self.verbose:
            selected = [i for i, v in enumerate(self.global_best_features) if v > 0.5]
            print(f"\nPSO优化完成！最佳适应度: {self.global_best_fitness:.6f}")
            print(f"最佳特征选择: {selected}")

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
        selected_indices = [i for i, v in enumerate(feature_selection) if v > 0.5]

        if len(selected_indices) == 0:
            return float('inf')

        # 解析超参数
        hidden_dim1 = int(np.clip(hyperparams[0], 16, 512)) if len(hyperparams) > 0 else 128
        hidden_dim2 = int(np.clip(hyperparams[1], 16, 256)) if len(hyperparams) > 1 else 64
        lr = np.clip(hyperparams[2], 1e-5, 0.1) if len(hyperparams) > 2 else 0.001
        batch_size = int(np.clip(hyperparams[3], 16, 256)) if len(hyperparams) > 3 else 64
        window_size = int(np.clip(hyperparams[4], 12, 96)) if len(hyperparams) > 4 else 48

        cv_rmse_list = []

        for fold in range(3):
            try:
                X_train_fold = X_train_list[fold][:, selected_indices]
                y_train_fold = y_train_list[fold]
                X_val_fold = X_val_list[fold][:, selected_indices]
                y_val_fold = y_val_list[fold]

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

            except Exception:
                continue

        if len(cv_rmse_list) == 0:
            return float('inf')

        avg_rmse = np.mean(cv_rmse_list)
        penalty = 0.001 * len(selected_indices)

        return avg_rmse + penalty

    return fitness_func
