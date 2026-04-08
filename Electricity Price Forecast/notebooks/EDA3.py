"""
EDA3 - 缺失值填充方法性能对比测试
测试 MLP 神经网络与线性插值法在电价数据缺失值填充上的性能

测试流程：
1. 截取未缺失的最长连续样本
2. 在样本中产生 1-2 天的随机连续缺失（缺失率<20%）
3. 使用不同填充器进行填充
4. 使用 RMSE 评估对比效果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error
import warnings

# 设置中文字体和绘图样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 第一部分：数据加载和预处理
# ============================================================================

def load_processed_data(data_path: str = None) -> pd.DataFrame:
    """加载处理后的数据"""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', '市场边界_出清价格总表.csv')

    print(f"加载处理后的数据：{data_path}")
    df = pd.read_csv(data_path)

    # 将时间戳列转换为datetime并设为索引
    if '时间戳' in df.columns:
        df['datetime'] = pd.to_datetime(df['时间戳'].str.split('_').str[0])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError("数据中未找到时间戳或datetime列")

    df.set_index('datetime', inplace=True)
    return df


def find_longest_continuous_segment(data: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, slice]:
    """
    查找最长的连续时间段（以目标列的非缺失值为基准）

    Parameters
    ----------
    data : pd.DataFrame
        数据框
    target_col : str
        目标列名，如果为None则使用第一列

    Returns
    -------
    segment : pd.DataFrame
        最长连续时间段的数据
    time_slice : slice
        时间切片位置
    """
    print("\n" + "=" * 60)
    print("查找最长连续时间段")
    print("=" * 60)

    # 确定目标列
    if target_col is None:
        # 查找日前价格列
        price_cols = [col for col in data.columns if '日前' in col and '价格' in col]
        if price_cols:
            target_col = price_cols[0]
        else:
            target_col = data.columns[0]

    print(f"以列 '{target_col}' 为基准查找最长连续段")

    # 获取目标列的非缺失值位置
    valid_mask = data[target_col].notna()
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(f"目标列 '{target_col}' 没有有效值")

    # 找到所有连续段
    segments = []
    start_idx = valid_indices[0]
    prev_idx = valid_indices[0]

    for idx in valid_indices[1:]:
        if idx != prev_idx + 1:
            # 不连续，结束当前段
            segments.append((start_idx, prev_idx + 1))
            start_idx = idx
        prev_idx = idx

    # 添加最后一个段
    segments.append((start_idx, prev_idx + 1))

    if len(segments) == 0:
        print(f"找到 1 个连续时间段")
        print(f"  1. 长度：{len(data)} 小时 ({len(data)/24:.1f} 天), 时间：{data.index[0]} 到 {data.index[-1]}")
        return data, slice(0, len(data))

    # 找到最长的段
    longest_segment = max(segments, key=lambda x: x[1] - x[0])

    print(f"找到 {len(segments)} 个连续时间段")
    print(f"前 5 个最长的时间段:")
    for i, (start, end) in enumerate(sorted(segments, key=lambda x: x[1] - x[0], reverse=True)[:5]):
        length = end - start
        print(f"  {i+1}. 长度：{length} 小时 ({length/24:.1f} 天), 时间：{data.index[start]} 到 {data.index[end-1]}")

    start_idx, end_idx = longest_segment
    segment = data.iloc[start_idx:end_idx].copy()

    print(f"\n选择最长连续段：{len(segment)} 小时 ({len(segment)/24:.1f} 天)")
    print(f"时间范围：{segment.index[0]} 到 {segment.index[-1]}")

    return segment, slice(start_idx, end_idx)


# ============================================================================
# 第二部分：缺失值生成
# ============================================================================

def generate_missing_values(data: np.ndarray, missing_rate: float = 0.2, 
                           min_gap: int = 24, max_gap: int = 48) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    生成随机连续缺失值
    
    Parameters
    ----------
    data : np.ndarray
        原始数据
    missing_rate : float
        目标缺失率
    min_gap : int
        最小缺失长度（小时）
    max_gap : int
        最大缺失长度（小时）
        
    Returns
    -------
    data_with_missing : np.ndarray
        带缺失值的数据
    missing_mask : np.ndarray
        缺失值掩码
    gap_starts : List[int]
        缺失段起始位置
    gap_ends : List[int]
        缺失段结束位置
    """
    n = len(data)
    max_missing = int(n * missing_rate)
    
    data_with_missing = data.copy()
    missing_mask = np.zeros(n, dtype=bool)
    
    gap_starts = []
    gap_ends = []
    
    current_missing = 0
    
    while current_missing < max_missing:
        # 随机生成本次缺失的长度
        remaining_capacity = max_missing - current_missing
        
        # 确保有足够的剩余容量
        if remaining_capacity < min_gap:
            break
        
        gap_length = np.random.randint(min_gap, min(max_gap, remaining_capacity) + 1)
        
        # 随机选择起始位置
        max_start = n - gap_length
        if max_start <= 0:
            break
            
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + gap_length
        
        # 检查是否与已有缺失重叠
        if not np.any(missing_mask[start_idx:end_idx]):
            # 设置缺失值
            data_with_missing[start_idx:end_idx] = np.nan
            missing_mask[start_idx:end_idx] = True
            
            gap_starts.append(start_idx)
            gap_ends.append(end_idx)
            
            current_missing += gap_length
    
    # 转换为 pandas Series 以便插值
    series = pd.Series(data_with_missing)
    
    # 对缺失段进行标记
    print(f"\n缺失值数量：{np.sum(missing_mask)} ({np.sum(missing_mask)/len(data)*100:.2f}%)")
    
    return data_with_missing, missing_mask, gap_starts, gap_ends


# ============================================================================
# 第三部分：填充器实现
# ============================================================================

class BaseImputer:
    """填充器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def fit(self, data: np.ndarray, missing_mask: np.ndarray):
        """训练填充器"""
        pass
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        raise NotImplementedError


class LinearInterpolationImputer(BaseImputer):
    """线性插值填充器"""
    
    def __init__(self):
        super().__init__("线性插值")
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        filled = pd.Series(data).interpolate(method='linear')
        return filled.values


class NeuralNetworkImputer(BaseImputer):
    """神经网络填充器（MLP）"""
    
    def __init__(self, window_size: int = 48, output_size: int = 24,
                 hidden_dims: List[int] = None,
                 epochs: int = 100, batch_size: int = 64, lr: float = 0.001):
        super().__init__("神经网络 (MLP)")
        self.window_size = window_size
        self.output_size = output_size
        # 更大的网络容量
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        
    def fit(self, data: np.ndarray, missing_mask: np.ndarray):
        """训练神经网络模型"""
        print(f"\n  训练神经网络模型 ({self.epochs}轮, lr={self.lr})...")
        
        # 添加项目根目录到路径
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from models.neural_networks import MLPModel
        except ImportError as e:
            print(f"  警告：无法导入 MLPModel，使用线性插值代替：{e}")
            return
        
        # 准备训练数据
        valid_data = data[~missing_mask]
        
        if len(valid_data) < self.window_size + self.output_size:
            print(f"  警告：有效数据不足")
            return
        
        # 构建训练样本
        X_list = []
        y_list = []
        
        for i in range(len(valid_data) - self.window_size - self.output_size + 1):
            X_list.append(valid_data[i:i + self.window_size])
            y_list.append(valid_data[i + self.window_size:i + self.window_size + self.output_size])
        
        if len(X_list) < 10:
            print(f"  警告：训练样本不足")
            return
        
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        
        # 创建并训练模型（不使用标准化，保持原始数据尺度）
        self.model = MLPModel(
            input_dim=self.window_size,
            output_dim=self.output_size,
            hidden_dims=self.hidden_dims,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            device='cuda'
        )
        
        self.model.fit(X_train, y_train)
        print(f"  神经网络模型训练完成 ({self.epochs}轮)")
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        """使用神经网络填充缺失值（支持反标准化）"""
        if self.model is None:
            print("  警告：神经网络模型未训练，使用线性插值代替")
            return LinearInterpolationImputer().impute(data)
        
        filled = data.copy()
        missing_mask = np.isnan(data)
        
        if not np.any(missing_mask):
            return filled
        
        # 找到所有缺失段
        gap_indices = np.where(missing_mask)[0]
        if len(gap_indices) == 0:
            return filled
        
        # 将缺失段分组为连续区间
        gap_groups = []
        current_group = [gap_indices[0]]
        
        for i in range(1, len(gap_indices)):
            if gap_indices[i] == gap_indices[i-1] + 1:
                current_group.append(gap_indices[i])
            else:
                gap_groups.append(current_group)
                current_group = [gap_indices[i]]
        
        gap_groups.append(current_group)
        
        # 对每个缺失段进行填充
        for gap_group in gap_groups:
            chunk_start = gap_group[0]
            chunk_end = gap_group[-1] + 1
            chunk_length = chunk_end - chunk_start
            
            # 使用滚动预测处理任意长度的缺失段
            current_pos = chunk_start
            while current_pos < chunk_end:
                # 计算本次预测的长度
                predict_length = min(self.output_size, chunk_end - current_pos)
                
                # 构建输入窗口
                if current_pos >= self.window_size:
                    input_window = filled[current_pos - self.window_size:current_pos]
                else:
                    input_window = np.pad(filled[:current_pos], 
                                        (self.window_size - current_pos, 0), 
                                        mode='edge')
                
                # 处理输入窗口中的 NaN（用线性插值临时填充）
                input_window = pd.Series(input_window).interpolate(method='linear').values
                input_window = np.nan_to_num(input_window, nan=0.0)
                
                # 预测
                try:
                    prediction = self.model.predict(input_window.reshape(1, -1))[0]
                    actual_length = min(predict_length, len(prediction))
                    filled[current_pos:current_pos + actual_length] = prediction[:actual_length]
                    current_pos += actual_length
                except Exception as e:
                    print(f"    预测失败，使用插值：{str(e)}")
                    # 预测失败时使用插值
                    filled[current_pos:chunk_end] = np.nanmean(filled)
                    break
        
        # 确保没有 NaN
        if np.any(np.isnan(filled)):
            nan_indices = np.where(np.isnan(filled))[0]
            print(f"  警告：填充后仍有 {len(nan_indices)} 个 NaN 值，使用均值填充")
            filled = pd.Series(filled).interpolate(method='linear').values
            filled = np.nan_to_num(filled, nan=np.nanmean(filled))
        
        return filled


class MultiFeatureMLPImputer(BaseImputer):
    """多特征MLP填充器（使用日前数据、时间特征等）"""
    
    def __init__(self, feature_cols: List[str], window_size: int = 48, output_size: int = 24,
                 hidden_dims: List[int] = None, epochs: int = 100, 
                 batch_size: int = 64, lr: float = 0.001):
        super().__init__("多特征MLP")
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.scaler = None
        
    def fit(self, data: np.ndarray, missing_mask: np.ndarray, features_df: pd.DataFrame = None):
        """
        训练多特征MLP模型
        
        Parameters
        ----------
        data : np.ndarray
            目标价格数据
        missing_mask : np.ndarray
            缺失值掩码
        features_df : pd.DataFrame
            特征数据框
        """
        if features_df is None or len(self.feature_cols) == 0:
            print("  警告：没有提供特征数据，使用单变量MLP")
            # 降级为单变量MLP
            simple_imputer = NeuralNetworkImputer(
                window_size=self.window_size,
                output_size=self.output_size,
                hidden_dims=self.hidden_dims,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr
            )
            simple_imputer.fit(data, missing_mask)
            self.model = simple_imputer.model
            return
        
        print(f"\n  训练多特征MLP模型 ({self.epochs}轮, 特征: {self.feature_cols})...")
        
        # 添加项目根目录到路径
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from models.neural_networks import MLPModel
        except ImportError as e:
            print(f"  警告：无法导入 MLPModel：{e}")
            return
        
        # 准备特征数据
        feature_data = features_df[self.feature_cols].values.astype(float)
        
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # 准备训练数据
        valid_indices = np.where(~missing_mask)[0]
        
        if len(valid_indices) < self.window_size + self.output_size + 10:
            print(f"  警告：有效数据不足")
            return
        
        # 构建训练样本：价格历史 + 当前特征
        X_list = []
        y_list = []
        
        for i in range(len(valid_indices) - self.window_size - self.output_size):
            idx = valid_indices[i]
            end_idx = valid_indices[i] + self.window_size
            
            # 检查窗口内是否连续且有效
            if end_idx >= len(data):
                continue
            if np.any(missing_mask[idx:end_idx]):
                continue
                
            # 价格历史
            price_history = data[idx:end_idx]
            # 当前时刻特征
            current_features = feature_data_scaled[end_idx - 1]
            
            # 合并输入
            combined_input = np.concatenate([price_history, current_features])
            X_list.append(combined_input)
            
            # 预测目标
            target_idx = end_idx
            if target_idx + self.output_size <= len(data):
                y_list.append(data[target_idx:target_idx + self.output_size])
        
        if len(X_list) < 10:
            print(f"  警告：训练样本不足 ({len(X_list)}个)")
            return
        
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        
        print(f"  训练样本数: {len(X_train)}, 输入维度: {X_train.shape[1]}")
        
        # 创建并训练模型
        self.model = MLPModel(
            input_dim=X_train.shape[1],
            output_dim=self.output_size,
            hidden_dims=self.hidden_dims,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            device='cuda'
        )
        
        self.model.fit(X_train, y_train)
        print(f"  多特征MLP模型训练完成 ({self.epochs}轮)")
    
    def impute(self, data: np.ndarray, features_df: pd.DataFrame = None) -> np.ndarray:
        """使用多特征MLP填充缺失值"""
        if self.model is None:
            print("  警告：模型未训练，使用线性插值代替")
            return LinearInterpolationImputer().impute(data)
        
        filled = data.copy()
        missing_mask = np.isnan(data)
        
        if not np.any(missing_mask):
            return filled
        
        # 准备特征数据
        if features_df is not None and self.scaler is not None:
            feature_data = features_df[self.feature_cols].values.astype(float)
            feature_data_scaled = self.scaler.transform(feature_data)
        else:
            feature_data_scaled = None
        
        # 找到所有缺失段
        gap_indices = np.where(missing_mask)[0]
        if len(gap_indices) == 0:
            return filled
        
        # 将缺失段分组为连续区间
        gap_groups = []
        current_group = [gap_indices[0]]
        
        for i in range(1, len(gap_indices)):
            if gap_indices[i] == gap_indices[i-1] + 1:
                current_group.append(gap_indices[i])
            else:
                gap_groups.append(current_group)
                current_group = [gap_indices[i]]
        
        gap_groups.append(current_group)
        
        # 对每个缺失段进行填充
        for gap_group in gap_groups:
            chunk_start = gap_group[0]
            chunk_end = gap_group[-1] + 1
            
            # 使用滚动预测处理任意长度的缺失段
            current_pos = chunk_start
            while current_pos < chunk_end:
                predict_length = min(self.output_size, chunk_end - current_pos)
                
                # 构建输入窗口
                if current_pos >= self.window_size:
                    input_window = filled[current_pos - self.window_size:current_pos]
                    if feature_data_scaled is not None and current_pos < len(feature_data_scaled):
                        current_features = feature_data_scaled[current_pos]
                    else:
                        current_features = np.zeros(len(self.feature_cols))
                else:
                    input_window = np.pad(filled[:current_pos], 
                                        (self.window_size - current_pos, 0), 
                                        mode='edge')
                    if feature_data_scaled is not None and current_pos < len(feature_data_scaled):
                        current_features = feature_data_scaled[current_pos]
                    else:
                        current_features = np.zeros(len(self.feature_cols))
                
                # 处理输入窗口中的 NaN
                input_window = pd.Series(input_window).interpolate(method='linear').values
                input_window = np.nan_to_num(input_window, nan=0.0)
                
                # 合并价格历史和特征
                combined_input = np.concatenate([input_window, current_features])
                
                # 预测
                try:
                    prediction = self.model.predict(combined_input.reshape(1, -1))[0]
                    actual_length = min(predict_length, len(prediction))
                    filled[current_pos:current_pos + actual_length] = prediction[:actual_length]
                    current_pos += actual_length
                except Exception as e:
                    print(f"    预测失败，使用插值：{str(e)}")
                    filled[current_pos:chunk_end] = np.nanmean(filled)
                    break
        
        # 确保没有 NaN
        if np.any(np.isnan(filled)):
            filled = pd.Series(filled).interpolate(method='linear').values
            filled = np.nan_to_num(filled, nan=np.nanmean(filled))
        
        return filled


class PSOMLPImputer(BaseImputer):
    """PSO优化MLP填充器"""
    
    def __init__(self, available_features: List[str],
                 n_particles: int = 15, max_iter: int = 30,
                 window_size: int = 48, output_size: int = 24,
                 epochs: int = 100):
        super().__init__("PSO-MLP")
        self.available_features = available_features
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.window_size = window_size
        self.output_size = output_size
        self.epochs = epochs
        self.model = None
        self.selected_features = None
        self.best_hyperparams = None
        self.scaler = None
        
    def fit(self, data: np.ndarray, missing_mask: np.ndarray, features_df: pd.DataFrame = None):
        """
        使用PSO优化特征选择和超参数
        
        Parameters
        ----------
        data : np.ndarray
            目标价格数据
        missing_mask : np.ndarray
            缺失值掩码
        features_df : pd.DataFrame
            特征数据框
        """
        if features_df is None:
            print("  警告：没有提供特征数据，使用默认MLP")
            simple_imputer = NeuralNetworkImputer(
                window_size=self.window_size,
                output_size=self.output_size,
                epochs=self.epochs
            )
            simple_imputer.fit(data, missing_mask)
            self.model = simple_imputer.model
            return
        
        # PSO参数：粒子数 = 特征数 + 超参数数 + 1
        print(f"\n  使用PSO优化MLP (粒子数: {self.n_particles}, 迭代: {self.max_iter})...")
        print(f"  可用特征: {self.available_features}")
        
        # 添加项目根目录到路径
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from models.pso_optimizer import PSOOptimizer
            from models.neural_networks import MLPModel
        except ImportError as e:
            print(f"  警告：无法导入优化模块：{e}")
            return
        
        # 准备数据
        valid_indices = np.where(~missing_mask)[0]
        n_features = len(self.available_features)
        
        # 构建3折交叉验证数据
        fold_size = len(valid_indices) // 3
        
        X_folds = []
        y_folds = []
        
        for fold in range(3):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < 2 else len(valid_indices)
            fold_indices = valid_indices[start_idx:end_idx]
            
            # 提取特征和目标
            fold_features = features_df.iloc[fold_indices][self.available_features].values.astype(float)
            fold_target = data[fold_indices]
            
            X_folds.append(fold_features)
            y_folds.append(fold_target)
        
        # PSO适应度函数
        def fitness_func(feature_selection, hyperparams):
            selected_indices = [i for i, v in enumerate(feature_selection) if v > 0.5]
            if len(selected_indices) == 0:
                return float('inf')
            
            selected_feature_names = [self.available_features[i] for i in selected_indices]
            
            # 解析超参数
            hidden_dim = int(np.clip(hyperparams[0], 32, 512))
            lr = np.clip(hyperparams[1], 1e-5, 0.01)
            batch_size = int(np.clip(hyperparams[2], 16, 128))
            
            # 3折交叉验证
            cv_rmse_list = []
            
            for val_fold in range(3):
                train_folds = [i for i in range(3) if i != val_fold]
                
                # 合并训练数据
                X_train = np.vstack([X_folds[i][:, selected_indices] for i in train_folds])
                y_train = np.concatenate([y_folds[i] for i in train_folds])
                X_val = X_folds[val_fold][:, selected_indices]
                y_val = y_folds[val_fold]
                
                if len(X_train) < self.window_size + 10 or len(X_val) < self.window_size:
                    continue
                
                try:
                    # 标准化
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # 构建序列数据
                    X_seq, y_seq = [], []
                    for i in range(len(X_train_scaled) - self.window_size):
                        # 价格历史 + 特征
                        price_hist = y_train[i:i+self.window_size]
                        features = X_train_scaled[i+self.window_size-1]
                        X_seq.append(np.concatenate([price_hist, features]))
                        y_seq.append(y_train[i+self.window_size])
                    
                    if len(X_seq) < 10:
                        continue
                    
                    X_seq = np.array(X_seq, dtype=np.float32)
                    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
                    
                    # PSO评估时训练
                    model = MLPModel(
                        input_dim=X_seq.shape[1],
                        output_dim=1,
                        hidden_dims=[hidden_dim, hidden_dim//2],
                        batch_size=batch_size,
                        epochs=self.epochs,
                        lr=lr,
                        device='cuda',
                        verbose=False
                    )
                    
                    model.fit(X_seq, y_seq)
                    
                    # 验证
                    X_val_seq, y_val_seq = [], []
                    for i in range(len(X_val_scaled) - self.window_size):
                        price_hist = y_val[i:i+self.window_size]
                        features = X_val_scaled[i+self.window_size-1]
                        X_val_seq.append(np.concatenate([price_hist, features]))
                        y_val_seq.append(y_val[i+self.window_size])
                    
                    if len(X_val_seq) == 0:
                        continue
                    
                    X_val_seq = np.array(X_val_seq, dtype=np.float32)
                    y_val_seq = np.array(y_val_seq, dtype=np.float32)
                    
                    predictions = model.predict(X_val_seq).flatten()
                    rmse = np.sqrt(mean_squared_error(y_val_seq, predictions))
                    cv_rmse_list.append(rmse)
                    
                except Exception as e:
                    continue
            
            if len(cv_rmse_list) == 0:
                return float('inf')
            
            avg_rmse = np.mean(cv_rmse_list)
            # 添加特征选择惩罚
            penalty = 0.0001 * len(selected_indices)
            return avg_rmse + penalty
        
        # 运行PSO优化
        # 超参数：hidden_dim, lr, batch_size
        hyperparam_bounds = [(32, 512), (1e-4, 0.01), (16, 128)]
        
        pso = PSOOptimizer(
            n_features=len(self.available_features),
            n_hyperparams=3,
            n_particles=self.n_particles,
            max_iter=self.max_iter,
            hyperparam_bounds=hyperparam_bounds,
            w=0.5, c1=1.5, c2=1.5,
            verbose=True
        )
        
        best_features, best_hyperparams, best_fitness = pso.optimize(fitness_func)
        
        # 保存最优解
        self.selected_features = [self.available_features[i] for i, v in enumerate(best_features) if v > 0.5]
        self.best_hyperparams = {
            'hidden_dim': int(np.clip(best_hyperparams[0], 32, 512)),
            'lr': np.clip(best_hyperparams[1], 1e-5, 0.01),
            'batch_size': int(np.clip(best_hyperparams[2], 16, 128))
        }
        
        print(f"\n  PSO优化完成！")
        print(f"  选中特征: {self.selected_features}")
        print(f"  最优超参数: {self.best_hyperparams}")
        print(f"  最佳适应度: {best_fitness:.6f}")
        
        # 使用最优特征和超参数训练最终模型
        self._train_final_model(data, missing_mask, features_df)
    
    def _train_final_model(self, data: np.ndarray, missing_mask: np.ndarray, features_df: pd.DataFrame):
        """使用最优参数训练最终模型"""
        from sklearn.preprocessing import StandardScaler
        
        # 准备特征数据
        feature_data = features_df[self.selected_features].values.astype(float)
        self.scaler = StandardScaler()
        feature_data_scaled = self.scaler.fit_transform(feature_data)
        
        # 准备训练数据
        valid_indices = np.where(~missing_mask)[0]
        
        X_list = []
        y_list = []
        
        for i in range(len(valid_indices) - self.window_size - self.output_size):
            idx = valid_indices[i]
            end_idx = valid_indices[i] + self.window_size
            
            if end_idx >= len(data) or np.any(missing_mask[idx:end_idx]):
                continue
            
            # 价格历史 + 特征
            price_history = data[idx:end_idx]
            current_features = feature_data_scaled[end_idx - 1]
            combined_input = np.concatenate([price_history, current_features])
            
            X_list.append(combined_input)
            y_list.append(data[end_idx:end_idx + self.output_size])
        
        if len(X_list) < 10:
            print("  警告：训练样本不足")
            return
        
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        
        print(f"  最终模型训练样本: {len(X_train)}")
        
        # 导入MLP模型
        from models.neural_networks import MLPModel
        
        # 创建并训练最终模型（PSO优化过程中不显示详细输出）
        self.model = MLPModel(
            input_dim=X_train.shape[1],
            output_dim=self.output_size,
            hidden_dims=[self.best_hyperparams['hidden_dim'], self.best_hyperparams['hidden_dim']//2],
            batch_size=self.best_hyperparams['batch_size'],
            epochs=self.epochs,
            lr=self.best_hyperparams['lr'],
            device='cuda',
            verbose=False
        )

        self.model.fit(X_train, y_train)
        print(f"  PSO-MLP最终模型训练完成 ({self.epochs}轮)")
    
    def impute(self, data: np.ndarray, features_df: pd.DataFrame = None) -> np.ndarray:
        """使用PSO-MLP填充缺失值"""
        if self.model is None:
            print("  警告：PSO-MLP模型未训练，使用线性插值代替")
            return LinearInterpolationImputer().impute(data)
        
        filled = data.copy()
        missing_mask = np.isnan(data)
        
        if not np.any(missing_mask):
            return filled
        
        # 准备特征数据
        if features_df is not None and self.scaler is not None and self.selected_features:
            feature_data = features_df[self.selected_features].values.astype(float)
            feature_data_scaled = self.scaler.transform(feature_data)
        else:
            feature_data_scaled = None
            if features_df is None:
                print("  警告：features_df为None")
            if self.scaler is None:
                print("  警告：scaler为None")
            if not self.selected_features:
                print("  警告：selected_features为空")
        
        # 找到所有缺失段
        gap_indices = np.where(missing_mask)[0]
        if len(gap_indices) == 0:
            return filled
        
        # 将缺失段分组
        gap_groups = []
        current_group = [gap_indices[0]]
        
        for i in range(1, len(gap_indices)):
            if gap_indices[i] == gap_indices[i-1] + 1:
                current_group.append(gap_indices[i])
            else:
                gap_groups.append(current_group)
                current_group = [gap_indices[i]]
        
        gap_groups.append(current_group)
        
        # 对每个缺失段进行填充
        for gap_group in gap_groups:
            chunk_start = gap_group[0]
            chunk_end = gap_group[-1] + 1
            
            current_pos = chunk_start
            while current_pos < chunk_end:
                predict_length = min(self.output_size, chunk_end - current_pos)
                
                # 构建输入窗口
                if current_pos >= self.window_size:
                    input_window = filled[current_pos - self.window_size:current_pos]
                    if feature_data_scaled is not None and current_pos < len(feature_data_scaled):
                        current_features = feature_data_scaled[current_pos]
                    elif feature_data_scaled is not None and len(feature_data_scaled) > 0:
                        # 如果超出范围，使用最后一个特征值
                        current_features = feature_data_scaled[-1]
                    else:
                        current_features = np.zeros(len(self.selected_features) if self.selected_features else 0)
                else:
                    input_window = np.pad(filled[:current_pos],
                                        (self.window_size - current_pos, 0),
                                        mode='edge')
                    if feature_data_scaled is not None and current_pos < len(feature_data_scaled):
                        current_features = feature_data_scaled[current_pos]
                    elif feature_data_scaled is not None and len(feature_data_scaled) > 0:
                        # 如果超出范围，使用最后一个特征值
                        current_features = feature_data_scaled[-1]
                    else:
                        current_features = np.zeros(len(self.selected_features) if self.selected_features else 0)
                
                # 处理输入窗口中的 NaN
                input_window = pd.Series(input_window).interpolate(method='linear').values
                input_window = np.nan_to_num(input_window, nan=0.0)
                
                # 合并价格历史和特征
                combined_input = np.concatenate([input_window, current_features])
                
                # 预测
                try:
                    prediction = self.model.predict(combined_input.reshape(1, -1))[0]
                    actual_length = min(predict_length, len(prediction))
                    filled[current_pos:current_pos + actual_length] = prediction[:actual_length]
                    current_pos += actual_length
                except Exception as e:
                    print(f"    预测失败，使用插值：{str(e)}")
                    filled[current_pos:chunk_end] = np.nanmean(filled)
                    break
        
        # 确保没有 NaN
        if np.any(np.isnan(filled)):
            filled = pd.Series(filled).interpolate(method='linear').values
            filled = np.nan_to_num(filled, nan=np.nanmean(filled))
        
        return filled


# ============================================================================
# 第四部分：评估指标和可视化
# ============================================================================

def calculate_rmse(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算 RMSE"""
    original_vals = original[mask]
    filled_vals = filled[mask]
    
    # 检查是否有 NaN
    nan_count = np.sum(np.isnan(filled_vals))
    if nan_count > 0:
        print(f"  警告：填充后的数据中有 {nan_count} 个 NaN 值")
        filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    
    rmse = np.sqrt(mean_squared_error(original_vals, filled_vals))
    return rmse


def calculate_mae(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算 MAE"""
    from sklearn.metrics import mean_absolute_error
    
    original_vals = original[mask]
    filled_vals = filled[mask]
    
    # 检查是否有 NaN
    nan_count = np.sum(np.isnan(filled_vals))
    if nan_count > 0:
        print(f"  警告：填充后的数据中有 {nan_count} 个 NaN 值")
        filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    
    mae = mean_absolute_error(original_vals, filled_vals)
    return mae


def calculate_smape(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算 sMAPE (Symmetric Mean Absolute Percentage Error)"""
    original_vals = original[mask]
    filled_vals = filled[mask]
    
    # 检查是否有 NaN
    nan_count = np.sum(np.isnan(filled_vals))
    if nan_count > 0:
        print(f"  警告：填充后的数据中有 {nan_count} 个 NaN 值")
        filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    
    # 避免除以 0
    denominator = (np.abs(original_vals) + np.abs(filled_vals)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape = np.mean(np.abs(filled_vals - original_vals) / denominator) * 100
    return smape


def evaluate_imputers(original_data: np.ndarray, data_with_missing: np.ndarray, 
                     missing_mask: np.ndarray, imputers: List[BaseImputer]) -> Dict:
    """
    评估不同填充器的性能
    """
    print("\n" + "=" * 60)
    print("评估填充器性能")
    print("=" * 60)
    
    results = {}
    
    for imputer in imputers:
        print(f"\n测试填充器：{imputer.name}")
        print("-" * 40)
        
        # 训练
        imputer.fit(data_with_missing, missing_mask)
        
        # 填充
        filled_data = imputer.impute(data_with_missing)
        
        # 评估
        rmse = calculate_rmse(original_data, filled_data, missing_mask)
        
        results[imputer.name] = {
            'rmse': rmse,
            'filled_data': filled_data
        }
        
        print(f"  RMSE: {rmse:.4f}")
    
    return results


def plot_rmse_comparison(results: Dict):
    """
    绘制 RMSE 柱状对比图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据
    names = list(results.keys())
    rmse_values = [results[name]['rmse'] for name in names]
    
    # 计算相对改进
    baseline_rmse = results['线性插值']['rmse']
    improvements = [(baseline_rmse - rmse) / baseline_rmse * 100 for rmse in rmse_values]
    
    # 创建柱状图
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色、青色、蓝色
    bars = ax.bar(names, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, rmse, imp) in enumerate(zip(bars, rmse_values, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'RMSE: {rmse:.2f}\n改进：{imp:+.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 设置标签和标题
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title('不同填充方法 RMSE 对比', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 旋转 x 轴标签
    plt.xticks(rotation=15, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(os.path.dirname(__file__), 'eda3_rmse_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRMSE 对比图已保存：{save_path}")
    
    plt.show()


def plot_imputation_comparison(original_data: np.ndarray, data_with_missing: np.ndarray,
                               missing_mask: np.ndarray, results: Dict, 
                               target_col: str = "电价"):
    """
    绘制填充效果对比图
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. 整体对比
    axes[0].plot(original_data, label='原始数据', linewidth=2, color='black', alpha=0.5)
    axes[0].plot(data_with_missing, label='带缺失值的数据', linewidth=1, color='gray', alpha=0.5)
    
    colors = {'线性插值': 'green', '神经网络 (MLP)': 'purple'}
    
    for name, result in results.items():
        if name in colors:
            axes[0].plot(result['filled_data'], label=name, alpha=0.7, linewidth=2, color=colors[name])
    
    axes[0].set_title('不同填充方法整体对比', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('时间点', fontsize=12)
    axes[0].set_ylabel(target_col, fontsize=12)
    
    # 2. 缺失区域详细对比
    if np.any(missing_mask):
        missing_indices = np.where(missing_mask)[0]
        
        # 选择几个典型的缺失区域进行展示
        gap_groups = []
        current_group = [missing_indices[0]]
        
        for i in range(1, len(missing_indices)):
            if missing_indices[i] == missing_indices[i-1] + 1:
                current_group.append(missing_indices[i])
            else:
                gap_groups.append(current_group)
                current_group = [missing_indices[i]]
        
        gap_groups.append(current_group)
        
        # 选择前 3 个缺失区域展示
        num_gaps_to_show = min(3, len(gap_groups))
        
        for gap_idx in range(num_gaps_to_show):
            gap_group = gap_groups[gap_idx]
            chunk_start = gap_group[0]
            chunk_end = gap_group[-1] + 1
            
            # 扩展显示范围
            display_start = max(0, chunk_start - 48)
            display_end = min(len(original_data), chunk_end + 48)
            
            # 创建子图
            ax_detail = plt.axes([0.1, 0.15 - gap_idx * 0.12, 0.8, 0.1])
            
            # 绘制原始数据
            ax_detail.plot(range(display_start, display_end), 
                          original_data[display_start:display_end],
                          label='原始数据', color='black', linewidth=2, alpha=0.7)
            
            # 绘制填充结果
            for name, result in results.items():
                if name in colors:
                    ax_detail.plot(range(display_start, display_end),
                                 result['filled_data'][display_start:display_end],
                                 label=name, alpha=0.8, linewidth=2, color=colors[name], linestyle='--')
            
            # 标记缺失区域
            ax_detail.axvspan(chunk_start, chunk_end, alpha=0.3, color='red', 
                            label='缺失区域' if gap_idx == 0 else None)
            
            ax_detail.set_title(f'缺失区域 {gap_idx + 1} 填充细节 (时间点 {chunk_start}-{chunk_end})',
                              fontsize=12, fontweight='bold')
            ax_detail.legend(loc='best', fontsize=9)
            ax_detail.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(os.path.dirname(__file__), 'eda3_imputation_details.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"填充效果对比图已保存：{save_path}")
    
    plt.show()


def plot_comparison(original_data: np.ndarray, data_with_missing: np.ndarray,
                   missing_mask: np.ndarray, results: Dict, 
                   target_col: str = "电价"):
    """
    绘制所有对比图
    """
    # 图 1: RMSE 柱状对比
    plot_rmse_comparison(results)
    
    # 图 2: 填充效果对比
    plot_imputation_comparison(original_data, data_with_missing, missing_mask, results, target_col)


# ============================================================================
# 第五部分：主测试流程
# ============================================================================

def run_test(data_path: str = None, target_col: str = "平均出清价格 - 日前（元/MWh）",
             missing_rate: float = 0.2):
    """
    运行完整的测试流程，对比三种方法：
    1. 线性插值（基线）
    2. MLP（单变量神经网络）
    3. PSO-MLP（PSO优化特征选择和超参数的多特征MLP）
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "缺失值填充方法性能对比测试")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n" + "=" * 60)
    print("数据加载")
    print("=" * 60)
    df = load_processed_data(data_path)
    print(f"数据形状：{df.shape}")
    print(f"时间范围：{df.index[0]} 到 {df.index[-1]}")
    
    # 2. 提取目标列和特征
    df_reset = df.reset_index()
    
    # 查找价格列
    price_cols = [col for col in df_reset.columns if '日前' in col and '价格' in col and '实时' not in col]
    if len(price_cols) == 0:
        print(f"错误：未找到日前价格列")
        print(f"可用列：{df_reset.columns.tolist()}")
        return
    
    target_col = price_cols[0]
    target_data = df_reset[target_col].values.astype(float)
    print(f"\n目标列：{target_col}")
    print(f"数据长度：{len(target_data)}")
    print(f"数据范围：{np.nanmin(target_data):.2f} 到 {np.nanmax(target_data):.2f}")
    
    # 3. 找到最长连续段（以日前价格为基准）
    segment, time_slice = find_longest_continuous_segment(df_reset, target_col)
    df_segment = df_reset.iloc[time_slice].copy()
    original_data = df_segment[target_col].values.astype(float)
    
    # 4. 准备特征（日前数据、小时、是否周末）
    print("\n" + "=" * 60)
    print("特征工程")
    print("=" * 60)
    
    # 定义候选特征：日前数据 + 小时 + 是否周末
    candidate_features = []

    # 时间特征：只选用小时和是否周末
    time_features = ['小时', '是否周末']
    for feat in time_features:
        if feat in df_segment.columns:
            candidate_features.append(feat)

    # 日前数据特征（系统负荷、非市场化机组出力）
    day_ahead_features = ['系统负荷-日前', '非市场化机组出力-日前']
    for feat in day_ahead_features:
        if feat in df_segment.columns:
            candidate_features.append(feat)

    # 新能源出力特征（风电、光伏、水电）
    new_energy_features = ['风电出力-日前', '光伏出力-日前', '水电出力-日前']
    for feat in new_energy_features:
        if feat in df_segment.columns:
            candidate_features.append(feat)
    
    print(f"候选特征：{candidate_features}")
    
    # 处理特征缺失值（使用线性插值填充）
    for feat in candidate_features:
        if feat in df_segment.columns:
            if df_segment[feat].isnull().any():
                n_missing = df_segment[feat].isnull().sum()
                print(f"  特征 '{feat}' 有 {n_missing} 个缺失值，使用线性插值填充")
                df_segment[feat] = df_segment[feat].interpolate(method='linear')
                # 如果还有缺失值（如开头或结尾），使用前向/后向填充
                df_segment[feat] = df_segment[feat].ffill().bfill()
    
    # 5. 生成缺失值
    print("\n" + "=" * 60)
    print("生成缺失值")
    print("=" * 60)
    np.random.seed(42)
    data_with_missing, missing_mask, gap_starts, gap_ends = generate_missing_values(
        original_data, missing_rate=missing_rate, min_gap=24, max_gap=48
    )
    
    # 6. 创建并评估填充器
    results = {}
    
    # 6.1 线性插值（基线）
    print("\n" + "=" * 60)
    print("1. 线性插值（基线）")
    print("=" * 60)
    linear_imputer = LinearInterpolationImputer()
    linear_imputer.fit(data_with_missing, missing_mask)
    linear_filled = linear_imputer.impute(data_with_missing)
    linear_rmse = calculate_rmse(original_data, linear_filled, missing_mask)
    results['线性插值'] = {'rmse': linear_rmse, 'filled_data': linear_filled}
    print(f"  RMSE: {linear_rmse:.4f}")
    
    # 6.2 MLP（单变量）
    print("\n" + "=" * 60)
    print("2. MLP（单变量神经网络）")
    print("=" * 60)
    mlp_imputer = NeuralNetworkImputer(
        window_size=48,
        output_size=24,
        hidden_dims=[256, 128, 64],
        epochs=100,
        batch_size=64,
        lr=0.001
    )
    mlp_imputer.fit(data_with_missing, missing_mask)
    mlp_filled = mlp_imputer.impute(data_with_missing)
    mlp_rmse = calculate_rmse(original_data, mlp_filled, missing_mask)
    results['MLP'] = {'rmse': mlp_rmse, 'filled_data': mlp_filled}
    print(f"  RMSE: {mlp_rmse:.4f}")
    
    # 6.3 PSO-MLP（PSO优化特征选择和超参数）
    print("\n" + "=" * 60)
    print("3. PSO-MLP（PSO优化特征选择和超参数）")
    print("=" * 60)
    pso_mlp_imputer = PSOMLPImputer(
        available_features=candidate_features,
        n_particles=15,
        max_iter=50,
        window_size=48,
        output_size=24,
        epochs=100
    )
    pso_mlp_imputer.fit(data_with_missing, missing_mask, df_segment)
    pso_mlp_filled = pso_mlp_imputer.impute(data_with_missing, df_segment)
    pso_mlp_rmse = calculate_rmse(original_data, pso_mlp_filled, missing_mask)
    results['PSO-MLP'] = {'rmse': pso_mlp_rmse, 'filled_data': pso_mlp_filled}
    print(f"  RMSE: {pso_mlp_rmse:.4f}")
    
    # 保存PSO优化结果信息
    if pso_mlp_imputer.selected_features is not None:
        results['PSO-MLP']['selected_features'] = pso_mlp_imputer.selected_features
        results['PSO-MLP']['hyperparams'] = pso_mlp_imputer.best_hyperparams
    
    # 7. 打印最终结果
    print("\n" + "=" * 80)
    print(" " * 30 + "最终评估结果")
    print("=" * 80)
    
    baseline_rmse = results['线性插值']['rmse']
    
    print(f"\n{'填充方法':<30} {'RMSE':>12} {'相对改进':>12}")
    print("-" * 60)
    
    # 按 RMSE 排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    
    for name, result in sorted_results:
        rmse = result['rmse']
        improvement = (baseline_rmse - rmse) / baseline_rmse * 100
        print(f"{name:<30} {rmse:>12.4f} {improvement:>11.2f}%")
    
    # 打印PSO优化详情
    if 'PSO-MLP' in results and 'selected_features' in results['PSO-MLP']:
        print("\n" + "=" * 80)
        print("PSO-MLP 优化详情")
        print("=" * 80)
        print(f"选中特征: {results['PSO-MLP']['selected_features']}")
        print(f"最优超参数:")
        for param, value in results['PSO-MLP']['hyperparams'].items():
            print(f"  {param}: {value}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return results


def run_test_with_features(data_path: str = None, 
                           target_col: str = "平均出清价格-日前（元/MWh）",
                           missing_rate: float = 0.2):
    """
    运行带特征工程的增强版测试流程
    使用多特征（价格、负荷、新能源出力、时间特征）进行填充
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "特征工程增强版缺失值填充测试")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n" + "=" * 60)
    print("数据加载与特征工程")
    print("=" * 60)
    df = load_processed_data(data_path)
    print(f"数据形状：{df.shape}")
    print(f"时间范围：{df.index[0]} 到 {df.index[-1]}")
    
    # 2. 提取特征列
    df_reset = df.reset_index()
    
    # 查找价格列
    price_cols = [col for col in df_reset.columns if '日前' in col and '价格' in col]
    if len(price_cols) == 0:
        print(f"错误：未找到日前价格列")
        print(f"可用列：{df_reset.columns.tolist()}")
        return
    
    target_col = price_cols[0]
    print(f"\n目标列：{target_col}")
    
    # 3. 找到最长连续段
    segment, time_slice = find_longest_continuous_segment(df)
    df_segment = df_reset.iloc[time_slice].copy()
    
    # 4. 特征工程 - 创建特征矩阵
    print("\n构建特征矩阵...")
    
    # 基础时间特征
    feature_cols = ['小时', '星期', '月', '是否周末', '是否高峰时段', '是否夜间', '季度']
    
    # 电力相关特征
    power_cols = ['系统负荷-日前', '非市场化机组出力-日前', '新能源出力-日前']
    
    # 检查哪些列存在
    available_features = []
    for col in feature_cols + power_cols:
        if col in df_segment.columns:
            available_features.append(col)
    
    print(f"使用特征：{available_features}")
    
    # 构建特征矩阵
    X_features = df_segment[available_features].values.astype(float)
    y_target = df_segment[target_col].values.astype(float)
    
    # 添加滞后特征（过去24小时的价格）
    print("添加滞后特征（过去24小时价格）...")
    lag_features = []
    for lag in [1, 2, 3, 6, 12, 24]:
        lag_col = f'price_lag_{lag}'
        df_segment[lag_col] = df_segment[target_col].shift(lag)
        lag_features.append(lag_col)
    
    # 添加滑动窗口统计特征
    print("添加滑动窗口统计特征...")
    df_segment['price_ma_6'] = df_segment[target_col].rolling(window=6, min_periods=1).mean()
    df_segment['price_ma_12'] = df_segment[target_col].rolling(window=12, min_periods=1).mean()
    df_segment['price_ma_24'] = df_segment[target_col].rolling(window=24, min_periods=1).mean()
    df_segment['price_std_24'] = df_segment[target_col].rolling(window=24, min_periods=1).std().fillna(0)
    
    stat_features = ['price_ma_6', 'price_ma_12', 'price_ma_24', 'price_std_24']
    
    # 合并所有特征
    all_feature_cols = available_features + lag_features + stat_features
    X_full = df_segment[all_feature_cols].fillna(0).values.astype(float)
    
    # 特征标准化 - 关键！防止NaN
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)
    
    print(f"特征矩阵形状：{X_full.shape}")
    print(f"特征列表：{all_feature_cols}")
    print(f"特征标准化完成（均值≈0，标准差≈1）")
    
    # 5. 生成缺失值（只在目标列上生成）
    print("\n" + "=" * 60)
    print("生成缺失值")
    print("=" * 60)
    np.random.seed(42)
    original_data = y_target.copy()
    data_with_missing, missing_mask, gap_starts, gap_ends = generate_missing_values(
        original_data, missing_rate=missing_rate, min_gap=24, max_gap=48
    )
    
    # 6. 使用带特征的MLP进行填充
    print("\n" + "=" * 60)
    print("使用特征增强MLP进行填充")
    print("=" * 60)
    
    filled_data = feature_enhanced_mlp_impute(
        data_with_missing, missing_mask, X_full,
        window_size=48, output_size=24,
        hidden_dims=[256, 128, 64, 32],
        epochs=100, batch_size=64, lr=0.001
    )
    
    # 7. 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    # 只在缺失位置计算误差
    missing_indices = np.where(missing_mask)[0]
    if len(missing_indices) > 0:
        y_true = original_data[missing_indices]
        y_pred = filled_data[missing_indices]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        print(f"\n特征增强MLP填充结果：")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  sMAPE: {smape:.2f}%")
        
        # 与线性插值对比
        linear_filled = LinearInterpolationImputer().impute(data_with_missing)
        linear_rmse = np.sqrt(mean_squared_error(original_data[missing_indices], 
                                                  linear_filled[missing_indices]))
        
        print(f"\n线性插值对比：")
        print(f"  线性插值 RMSE: {linear_rmse:.4f}")
        print(f"  改进幅度: {(linear_rmse - rmse) / linear_rmse * 100:.2f}%")
    
    return filled_data


def feature_enhanced_mlp_impute(data: np.ndarray, missing_mask: np.ndarray, 
                                features: np.ndarray,
                                window_size: int = 48, output_size: int = 24,
                                hidden_dims: List[int] = [256, 128, 64, 32],
                                epochs: int = 100, batch_size: int = 64,
                                lr: float = 0.001) -> np.ndarray:
    """
    使用特征增强的MLP进行缺失值填充
    """
    import sys
    import os
    project_root = os.path.join(os.path.dirname(__file__), '..')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from models.neural_networks import MLPModel
    except ImportError as e:
        print(f"警告：无法导入 MLPModel：{e}")
        return LinearInterpolationImputer().impute(data)
    
    filled = data.copy()
    valid_data = data[~missing_mask]
    
    if len(valid_data) < window_size + output_size:
        print("警告：有效数据不足，使用线性插值")
        return LinearInterpolationImputer().impute(data)
    
    # 构建训练样本（包含特征）
    X_list = []
    y_list = []
    
    valid_indices = np.where(~missing_mask)[0]
    
    for i in range(len(valid_indices) - window_size - output_size + 1):
        idx_start = valid_indices[i]
        idx_end = valid_indices[i] + window_size
        
        # 检查窗口内是否都是有效数据
        if idx_end > len(data):
            continue
        window_mask = missing_mask[idx_start:idx_end]
        if np.any(window_mask):
            continue
        
        # 价格历史 + 当前特征
        price_history = data[idx_start:idx_end]
        current_features = features[idx_end - 1]  # 窗口最后一个时刻的特征
        
        # 合并特征：价格历史 + 外部特征
        combined_input = np.concatenate([price_history, current_features])
        X_list.append(combined_input)
        
        # 预测目标：接下来的output_size个价格
        target_idx = valid_indices[i] + window_size
        if target_idx + output_size <= len(data):
            y_list.append(data[target_idx:target_idx + output_size])
    
    if len(X_list) < 10:
        print("警告：训练样本不足，使用线性插值")
        return LinearInterpolationImputer().impute(data)
    
    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)
    
    # 检查并处理NaN/Inf
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print(f"警告：训练数据中有NaN/Inf，进行清理...")
        print(f"  NaN数量: {np.sum(np.isnan(X_train))}")
        print(f"  Inf数量: {np.sum(np.isinf(X_train))}")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        print(f"警告：训练目标中有NaN/Inf，进行清理...")
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 对价格历史部分进行标准化（前window_size列是价格）
    price_mean = np.mean(X_train[:, :window_size])
    price_std = np.std(X_train[:, :window_size]) + 1e-8
    X_train[:, :window_size] = (X_train[:, :window_size] - price_mean) / price_std
    
    # 对目标也进行标准化
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    
    print(f"训练样本数：{len(X_train)}")
    print(f"输入维度：{X_train.shape[1]} (价格历史{window_size} + 特征{X_train.shape[1]-window_size})")
    print(f"价格标准化: mean={price_mean:.2f}, std={price_std:.2f}")
    print(f"目标标准化: mean={y_mean:.2f}, std={y_std:.2f}")
    
    # 创建并训练模型
    model = MLPModel(
        input_dim=X_train.shape[1],
        output_dim=output_size,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device='cuda'
    )
    
    model.fit(X_train, y_train_norm)
    print(f"特征增强MLP训练完成 ({epochs}轮)")
    
    # 保存标准化参数用于预测
    norm_params = {
        'price_mean': price_mean,
        'price_std': price_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    # 填充缺失值
    gap_indices = np.where(missing_mask)[0]
    if len(gap_indices) == 0:
        return filled
    
    # 将缺失段分组
    gap_groups = []
    current_group = [gap_indices[0]]
    
    for i in range(1, len(gap_indices)):
        if gap_indices[i] == gap_indices[i-1] + 1:
            current_group.append(gap_indices[i])
        else:
            gap_groups.append(current_group)
            current_group = [gap_indices[i]]
    gap_groups.append(current_group)
    
    # 对每个缺失段进行填充
    for gap_group in gap_groups:
        chunk_start = gap_group[0]
        chunk_end = gap_group[-1] + 1
        chunk_length = chunk_end - chunk_start
        
        current_pos = chunk_start
        while current_pos < chunk_end:
            predict_length = min(output_size, chunk_end - current_pos)
            
            # 构建输入
            if current_pos >= window_size:
                price_window = filled[current_pos - window_size:current_pos]
                current_feat = features[current_pos - 1]  # 使用当前时刻的特征
            else:
                price_window = np.pad(filled[:current_pos], 
                                    (window_size - current_pos, 0), 
                                    mode='edge')
                current_feat = features[current_pos] if current_pos < len(features) else features[0]
            
            # 处理价格窗口中的NaN
            price_window = pd.Series(price_window).interpolate(method='linear').values
            price_window = np.nan_to_num(price_window, nan=np.nanmean(filled))
            
            # 对价格窗口进行标准化
            price_window_norm = (price_window - norm_params['price_mean']) / norm_params['price_std']
            
            # 合并输入
            combined_input = np.concatenate([price_window_norm, current_feat])
            
            # 预测
            try:
                prediction_norm = model.predict(combined_input.reshape(1, -1))[0]
                # 反标准化
                prediction = prediction_norm * norm_params['y_std'] + norm_params['y_mean']
                actual_length = min(predict_length, len(prediction))
                filled[current_pos:current_pos + actual_length] = prediction[:actual_length]
                current_pos += actual_length
            except Exception as e:
                print(f"预测失败，使用插值：{str(e)}")
                filled[current_pos:chunk_end] = np.nanmean(filled)
                break
    
    # 确保没有NaN
    if np.any(np.isnan(filled)):
        filled = pd.Series(filled).interpolate(method='linear').values
        filled = np.nan_to_num(filled, nan=np.nanmean(filled))
    
    return filled


if __name__ == "__main__":
    import sys

    run_test()
