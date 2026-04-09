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
    """神经网络填充器（MLP），使用价格历史+小时特征"""
    
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
        self.hour_feature = None
        
    def fit(self, data: np.ndarray, missing_mask: np.ndarray, hour_feature: np.ndarray = None,
            original_data: np.ndarray = None):
        """
        训练神经网络模型
        
        Parameters
        ----------
        data : np.ndarray
            目标价格数据（包含缺失值，用于构建输入特征）
        missing_mask : np.ndarray
            缺失值掩码
        hour_feature : np.ndarray
            小时特征（0-23）
        original_data : np.ndarray, optional
            原始完整数据（不包含缺失值，用于训练目标）
        """
        print(f"\n  训练神经网络模型 ({self.epochs}轮, lr={self.lr})...")
        
        # 保存小时特征
        self.hour_feature = hour_feature
        
        # 添加项目根目录到路径
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from models.neural_networks import MLPModel
        except ImportError as e:
            print(f"  警告：无法导入 MLPModel，使用线性插值代替：{e}")
            return
        
        # 使用原始完整数据作为训练目标，如果没有则使用线性插值填充
        if original_data is not None:
            train_target = original_data
            # 对于输入特征，使用线性插值填充缺失值
            input_data = pd.Series(data).interpolate(method='linear').values
            input_data = np.nan_to_num(input_data, nan=np.nanmean(input_data))
        else:
            # 如果没有提供原始数据，使用插值填充
            train_target = pd.Series(data).interpolate(method='linear').values
            train_target = np.nan_to_num(train_target, nan=np.nanmean(train_target))
            input_data = train_target
        
        # 计算价格特征的统计信息用于归一化
        self.price_mean = np.mean(input_data)
        self.price_std = np.std(input_data)
        if self.price_std == 0:
            self.price_std = 1.0
        
        valid_indices = np.where(~missing_mask)[0]
        
        if len(valid_indices) < self.window_size + self.output_size:
            print(f"  警告：有效数据不足")
            return
        
        # 构建训练样本：价格历史 + 小时特征
        X_list = []
        y_list = []
        
        for i in range(len(valid_indices) - self.window_size - self.output_size + 1):
            idx = valid_indices[i]
            end_idx = idx + self.window_size
            
            # 检查窗口内是否连续且有效
            if end_idx >= len(data):
                continue
            if np.any(missing_mask[idx:end_idx]):
                continue
            
            # 检查目标区域是否有缺失值
            target_end = end_idx + self.output_size
            if target_end > len(data) or np.any(missing_mask[end_idx:target_end]):
                continue
            
            # 价格历史（使用插值填充后的数据作为输入，并进行标准化）
            price_history = input_data[idx:end_idx]
            price_history_normalized = (price_history - self.price_mean) / self.price_std
            
            # 当前时刻小时特征（cos/sin循环编码）
            if hour_feature is not None and end_idx < len(hour_feature):
                hour = float(hour_feature[end_idx]) % 24
                # 循环编码：cos和sin
                hour_cos = np.cos(2 * np.pi * hour / 24)
                hour_sin = np.sin(2 * np.pi * hour / 24)
                # 合并价格历史和小时特征
                combined_input = np.concatenate([price_history_normalized, [hour_cos, hour_sin]])
            else:
                combined_input = price_history_normalized
            
            X_list.append(combined_input)
            # 使用原始完整数据作为目标
            y_list.append(train_target[end_idx:target_end])
        
        if len(X_list) < 10:
            print(f"  警告：训练样本不足")
            return
        
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        
        # 检查是否有NaN
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            print(f"  警告：训练数据中存在NaN，进行清理...")
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
        
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
        print(f"  神经网络模型训练完成 ({self.epochs}轮)")
    
    def impute(self, data: np.ndarray, hour_feature: np.ndarray = None) -> np.ndarray:
        """
        使用神经网络填充缺失值
        
        Parameters
        ----------
        data : np.ndarray
            待填充数据
        hour_feature : np.ndarray
            小时特征（0-23），如果为None则使用fit时传入的特征
        """
        if self.model is None:
            print("  警告：神经网络模型未训练，使用线性插值代替")
            return LinearInterpolationImputer().impute(data)
        
        # 使用传入的小时特征或fit时保存的特征
        if hour_feature is None:
            hour_feature = self.hour_feature
        
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
                
                # 对输入窗口进行标准化（使用训练时的统计信息）
                input_window_normalized = (input_window - self.price_mean) / self.price_std
                
                # 添加小时特征（cos/sin循环编码）
                if hour_feature is not None and current_pos < len(hour_feature):
                    hour = float(hour_feature[current_pos]) % 24
                    # 循环编码：cos和sin
                    hour_cos = np.cos(2 * np.pi * hour / 24)
                    hour_sin = np.sin(2 * np.pi * hour / 24)
                    combined_input = np.concatenate([input_window_normalized, [hour_cos, hour_sin]])
                else:
                    combined_input = input_window_normalized
                
                # 预测
                try:
                    prediction = self.model.predict(combined_input.reshape(1, -1))[0]
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





class OptimizedMLPImputer(BaseImputer):
    """超参数优化的MLP填充器（使用网格搜索），使用价格历史+小时特征"""
    
    def __init__(self, window_size: int = 48, output_size: int = 24,
                 epochs: int = 100, n_trials: int = 10):
        super().__init__("优化MLP")
        self.window_size = window_size
        self.output_size = output_size
        self.epochs = epochs
        self.n_trials = n_trials
        self.model = None
        self.best_params = None
        self.hour_feature = None
        
    def fit(self, data: np.ndarray, missing_mask: np.ndarray, hour_feature: np.ndarray = None,
            original_data: np.ndarray = None):
        """
        使用网格搜索优化MLP超参数
        
        Parameters
        ----------
        data : np.ndarray
            目标价格数据（包含缺失值，用于构建输入特征）
        missing_mask : np.ndarray
            缺失值掩码
        hour_feature : np.ndarray
            小时特征（0-23）
        original_data : np.ndarray, optional
            原始完整数据（不包含缺失值，用于训练目标）
        """
        print(f"\n  使用网格搜索优化MLP超参数 ({self.n_trials}次试验)...")
        
        # 保存小时特征
        self.hour_feature = hour_feature
        
        # 添加项目根目录到路径
        project_root = os.path.join(os.path.dirname(__file__), '..')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from models.neural_networks import MLPModel
        except ImportError as e:
            print(f"  警告：无法导入 MLPModel：{e}")
            return
        
        # 使用原始完整数据作为训练目标，如果没有则使用线性插值填充
        if original_data is not None:
            train_target = original_data
            # 对于输入特征，使用线性插值填充缺失值
            input_data = pd.Series(data).interpolate(method='linear').values
            input_data = np.nan_to_num(input_data, nan=np.nanmean(input_data))
        else:
            # 如果没有提供原始数据，使用插值填充
            train_target = pd.Series(data).interpolate(method='linear').values
            train_target = np.nan_to_num(train_target, nan=np.nanmean(train_target))
            input_data = train_target
        
        # 计算价格特征的统计信息用于归一化
        self.price_mean = np.mean(input_data)
        self.price_std = np.std(input_data)
        if self.price_std == 0:
            self.price_std = 1.0
        
        valid_indices = np.where(~missing_mask)[0]
        
        if len(valid_indices) < self.window_size + self.output_size + 50:
            print(f"  警告：有效数据不足，跳过优化")
            return
        
        # 构建训练样本：价格历史 + 小时特征
        X_list = []
        y_list = []
        
        for i in range(len(valid_indices) - self.window_size - self.output_size + 1):
            idx = valid_indices[i]
            end_idx = idx + self.window_size
            
            # 检查窗口内是否连续且有效
            if end_idx >= len(data):
                continue
            if np.any(missing_mask[idx:end_idx]):
                continue
            
            # 检查目标区域是否有缺失值
            target_end = end_idx + self.output_size
            if target_end > len(data) or np.any(missing_mask[end_idx:target_end]):
                continue
            
            # 价格历史（使用插值填充后的数据作为输入，并进行标准化）
            price_history = input_data[idx:end_idx]
            price_history_normalized = (price_history - self.price_mean) / self.price_std
            
            # 当前时刻小时特征（cos/sin循环编码）
            if hour_feature is not None and end_idx < len(hour_feature):
                hour = float(hour_feature[end_idx]) % 24
                # 循环编码：cos和sin
                hour_cos = np.cos(2 * np.pi * hour / 24)
                hour_sin = np.sin(2 * np.pi * hour / 24)
                # 合并价格历史和小时特征
                combined_input = np.concatenate([price_history_normalized, [hour_cos, hour_sin]])
            else:
                combined_input = price_history_normalized
            
            X_list.append(combined_input)
            y_list.append(train_target[end_idx:target_end])
        
        if len(X_list) < 50:
            print(f"  警告：训练样本不足")
            return
        
        # 划分训练集和验证集
        n_samples = len(X_list)
        train_size = int(0.8 * n_samples)
        
        X_train = np.array(X_list[:train_size], dtype=np.float32)
        y_train = np.array(y_list[:train_size], dtype=np.float32)
        X_val = np.array(X_list[train_size:], dtype=np.float32)
        y_val = np.array(y_list[train_size:], dtype=np.float32)
        
        # 检查并清理NaN
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            print(f"  警告：训练数据中存在NaN，进行清理...")
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0)
        
        input_dim = X_train.shape[1]
        print(f"  训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}, 输入维度: {input_dim}")
        
        # 定义超参数搜索空间
        param_grid = {
            'hidden_dims': [
                [128, 64],
                [256, 128],
                [256, 128, 64],
                [512, 256, 128],
            ],
            'lr': [0.001, 0.0005, 0.0001],
            'batch_size': [32, 64, 128]
        }
        
        # 生成所有参数组合
        import itertools
        all_params = []
        for hidden_dims in param_grid['hidden_dims']:
            for lr in param_grid['lr']:
                for batch_size in param_grid['batch_size']:
                    all_params.append({
                        'hidden_dims': hidden_dims,
                        'lr': lr,
                        'batch_size': batch_size
                    })
        
        # 随机选择n_trials个参数组合
        np.random.seed(42)
        if len(all_params) > self.n_trials:
            selected_indices = np.random.choice(len(all_params), self.n_trials, replace=False)
            trial_params = [all_params[i] for i in selected_indices]
        else:
            trial_params = all_params
        
        print(f"  测试 {len(trial_params)} 组超参数...")
        
        # 评估每个参数组合
        best_rmse = float('inf')
        best_params = None
        
        for i, params in enumerate(trial_params):
            try:
                # 创建模型
                model = MLPModel(
                    input_dim=input_dim,
                    output_dim=self.output_size,
                    hidden_dims=params['hidden_dims'],
                    batch_size=params['batch_size'],
                    epochs=self.epochs,
                    lr=params['lr'],
                    device='cuda',
                    verbose=False
                )
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 验证
                predictions = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                
                print(f"    试验 {i+1}/{len(trial_params)}: hidden_dims={params['hidden_dims']}, "
                      f"lr={params['lr']}, batch_size={params['batch_size']} -> RMSE={rmse:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
                    
            except Exception as e:
                print(f"    试验 {i+1}/{len(trial_params)} 失败: {str(e)}")
                continue
        
        if best_params is None:
            print("  警告：所有试验都失败，使用默认参数")
            best_params = {
                'hidden_dims': [256, 128, 64],
                'lr': 0.001,
                'batch_size': 64
            }
        
        self.best_params = best_params
        print(f"\n  最优超参数:")
        print(f"    hidden_dims: {best_params['hidden_dims']}")
        print(f"    lr: {best_params['lr']}")
        print(f"    batch_size: {best_params['batch_size']}")
        print(f"    验证集RMSE: {best_rmse:.4f}")
        
        # 使用最优参数训练最终模型（使用全部数据）
        X_full = np.array(X_list, dtype=np.float32)
        y_full = np.array(y_list, dtype=np.float32)
        
        self.model = MLPModel(
            input_dim=input_dim,
            output_dim=self.output_size,
            hidden_dims=best_params['hidden_dims'],
            batch_size=best_params['batch_size'],
            epochs=self.epochs,
            lr=best_params['lr'],
            device='cuda',
            verbose=False
        )
        
        self.model.fit(X_full, y_full)
        print(f"  优化MLP模型训练完成 ({self.epochs}轮)")
    
    def impute(self, data: np.ndarray, hour_feature: np.ndarray = None) -> np.ndarray:
        """
        使用优化后的MLP填充缺失值
        
        Parameters
        ----------
        data : np.ndarray
            待填充数据
        hour_feature : np.ndarray
            小时特征（0-23），如果为None则使用fit时传入的特征
        """
        if self.model is None:
            print("  警告：优化MLP模型未训练，使用线性插值代替")
            return LinearInterpolationImputer().impute(data)
        
        # 使用传入的小时特征或fit时保存的特征
        if hour_feature is None:
            hour_feature = self.hour_feature
        
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
                
                # 对输入窗口进行标准化（使用训练时的统计信息）
                input_window_normalized = (input_window - self.price_mean) / self.price_std
                
                # 添加小时特征（cos/sin循环编码）
                if hour_feature is not None and current_pos < len(hour_feature):
                    hour = float(hour_feature[current_pos]) % 24
                    # 循环编码：cos和sin
                    hour_cos = np.cos(2 * np.pi * hour / 24)
                    hour_sin = np.sin(2 * np.pi * hour / 24)
                    combined_input = np.concatenate([input_window_normalized, [hour_cos, hour_sin]])
                else:
                    combined_input = input_window_normalized
                
                # 预测
                try:
                    prediction = self.model.predict(combined_input.reshape(1, -1))[0]
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


def plot_three_methods_comparison(test_data: np.ndarray, test_missing_mask: np.ndarray,
                                  results: Dict, target_col: str = "电价"):
    """
    绘制三张子图对比：线性插值、MLP、优化MLP
    每张子图显示实际值和填充值
    """
    # 选择测试集中第一个缺失区域进行展示
    missing_indices = np.where(test_missing_mask)[0]
    
    if len(missing_indices) == 0:
        print("测试集中没有缺失值，无法绘制对比图")
        return
    
    # 找到第一个缺失区域
    gap_start = missing_indices[0]
    gap_end = missing_indices[0]
    for i in range(1, len(missing_indices)):
        if missing_indices[i] == missing_indices[i-1] + 1:
            gap_end = missing_indices[i]
        else:
            break
    
    # 扩展显示范围（前后各48小时）
    display_start = max(0, gap_start - 48)
    display_end = min(len(test_data), gap_end + 48 + 1)
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    methods = ['线性插值', 'MLP', '优化MLP']
    colors = {'实际值': 'black', '填充值': 'red'}
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # 绘制实际值
        ax.plot(range(display_start, display_end), 
                test_data[display_start:display_end],
                label='实际值', color=colors['实际值'], linewidth=2, alpha=0.8)
        
        # 绘制填充值
        if method in results:
            filled_data = results[method]['filled_data']
            ax.plot(range(display_start, display_end),
                   filled_data[display_start:display_end],
                   label='填充值', color=colors['填充值'], linewidth=2, alpha=0.8, linestyle='--')
        
        # 标记缺失区域
        ax.axvspan(gap_start, gap_end + 1, alpha=0.2, color='blue', label='缺失区域')
        
        # 计算该区域的RMSE
        if method in results:
            region_mask = np.zeros_like(test_missing_mask, dtype=bool)
            region_mask[gap_start:gap_end+1] = True
            region_rmse = calculate_rmse(test_data, results[method]['filled_data'], region_mask)
            title = f'{method} (区域RMSE: {region_rmse:.2f})'
        else:
            title = method
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(target_col, fontsize=12)
        
        # 只在最后一个子图显示x轴标签
        if idx == 2:
            ax.set_xlabel('时间点', fontsize=12)
    
    plt.suptitle('三种填充方法效果对比（测试集）', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(os.path.dirname(__file__), 'eda3_three_methods_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n三种方法对比图已保存：{save_path}")
    
    plt.show()


# ============================================================================
# 第五部分：主测试流程（严格划分训练/验证/测试集）
# ============================================================================

def split_train_val_test(data: np.ndarray, hour_feature: np.ndarray, 
                         train_ratio=0.6, val_ratio=0.2):
    """
    严格按时间顺序划分训练/验证/测试集
    
    Returns
    -------
    train_data, val_data, test_data : 划分后的数据和索引
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    train_hour = hour_feature[:train_end]
    train_indices = np.arange(0, train_end)
    
    val_data = data[train_end:val_end]
    val_hour = hour_feature[train_end:val_end]
    val_indices = np.arange(train_end, val_end)
    
    test_data = data[val_end:]
    test_hour = hour_feature[val_end:]
    test_indices = np.arange(val_end, n)
    
    print(f"  数据集划分:")
    print(f"    训练集: {len(train_data)} 样本 (索引 0-{train_end-1})")
    print(f"    验证集: {len(val_data)} 样本 (索引 {train_end}-{val_end-1})")
    print(f"    测试集: {len(test_data)} 样本 (索引 {val_end}-{n-1})")
    
    return (train_data, train_hour, train_indices), (val_data, val_hour, val_indices), (test_data, test_hour, test_indices)


def run_test(data_path: str = None, target_col: str = "平均出清价格 - 日前（元/MWh）",
             missing_rate: float = 0.2):
    """
    运行完整的测试流程，严格划分训练/验证/测试集：
    1. 训练集：训练模型参数
    2. 验证集：选择超参数（仅优化MLP使用）
    3. 测试集：评估最终性能（所有方法使用）
    
    对比三种方法：
    1. 线性插值（基线）
    2. MLP（单变量神经网络）
    3. 优化MLP（网格搜索超参数的MLP）
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "缺失值填充方法性能对比测试")
    print("=" * 80)
    print("\n【严格划分】训练集(60%) -> 验证集(20%) -> 测试集(20%)")
    
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
    
    # 4. 提取小时特征
    print("\n" + "=" * 60)
    print("特征工程")
    print("=" * 60)
    
    # 从数据中提取小时特征
    if '小时' in df_segment.columns:
        hour_feature = df_segment['小时'].values
    elif hasattr(df_segment.index, 'hour'):
        hour_feature = df_segment.index.hour.values
    elif 'datetime' in df_segment.columns:
        hour_feature = pd.to_datetime(df_segment['datetime']).dt.hour.values
    elif '时间戳' in df_segment.columns:
        hour_feature = pd.to_datetime(df_segment['时间戳'].str.split('_').str[0]).dt.hour.values
    else:
        hour_feature = np.arange(len(df_segment)) % 24
        print("  警告：未找到时间列，使用循环小时特征（0-23）")
    
    print(f"  小时特征范围: {hour_feature.min()}-{hour_feature.max()}")
    
    # 5. 严格划分训练/验证/测试集
    print("\n" + "=" * 60)
    print("数据集划分（按时间顺序）")
    print("=" * 60)
    (train_data, train_hour, train_indices), (val_data, val_hour, val_indices), (test_data, test_hour, test_indices) = \
        split_train_val_test(original_data, hour_feature, train_ratio=0.6, val_ratio=0.2)
    
    # 6. 在训练集上生成缺失值并训练模型
    print("\n" + "=" * 60)
    print("训练阶段：在训练集上生成缺失值并训练模型")
    print("=" * 60)
    np.random.seed(42)
    train_data_missing, train_missing_mask, _, _ = generate_missing_values(
        train_data, missing_rate=missing_rate, min_gap=24, max_gap=48
    )
    
    # 6.1 在训练集上训练MLP
    print("\n  训练MLP模型...")
    mlp_imputer = NeuralNetworkImputer(
        window_size=48,
        output_size=24,
        hidden_dims=[256, 128, 64],
        epochs=100,
        batch_size=64,
        lr=0.001
    )
    mlp_imputer.fit(train_data_missing, train_missing_mask, train_hour, train_data)
    
    # 6.2 在训练集+验证集上训练优化MLP（使用验证集选择超参数）
    print("\n  训练优化MLP模型（使用验证集选择超参数）...")
    # 合并训练集和验证集用于超参数搜索
    train_val_data = np.concatenate([train_data, val_data])
    train_val_hour = np.concatenate([train_hour, val_hour])
    train_val_missing, train_val_mask, _, _ = generate_missing_values(
        train_val_data, missing_rate=missing_rate, min_gap=24, max_gap=48
    )
    
    opt_mlp_imputer = OptimizedMLPImputer(
        window_size=48,
        output_size=24,
        epochs=100,
        n_trials=36  # 减少试验次数以加快训练
    )
    opt_mlp_imputer.fit(train_val_data, train_val_mask, train_val_hour, train_val_data)
    
    # 7. 在测试集上评估所有方法
    print("\n" + "=" * 60)
    print("测试阶段：在测试集上评估所有方法")
    print("=" * 60)
    np.random.seed(123)  # 使用不同的随机种子
    test_data_missing, test_missing_mask, _, _ = generate_missing_values(
        test_data, missing_rate=missing_rate, min_gap=24, max_gap=48
    )
    
    results = {}
    
    # 7.1 线性插值（基线）
    print("\n" + "=" * 60)
    print("1. 线性插值（基线）")
    print("=" * 60)
    linear_imputer = LinearInterpolationImputer()
    linear_imputer.fit(test_data_missing, test_missing_mask)
    linear_filled = linear_imputer.impute(test_data_missing)
    linear_rmse = calculate_rmse(test_data, linear_filled, test_missing_mask)
    results['线性插值'] = {'rmse': linear_rmse, 'filled_data': linear_filled}
    print(f"  测试集RMSE: {linear_rmse:.4f}")
    
    # 7.2 MLP
    print("\n" + "=" * 60)
    print("2. MLP（价格历史+小时特征）")
    print("=" * 60)
    mlp_filled = mlp_imputer.impute(test_data_missing, test_hour)
    mlp_rmse = calculate_rmse(test_data, mlp_filled, test_missing_mask)
    results['MLP'] = {'rmse': mlp_rmse, 'filled_data': mlp_filled}
    print(f"  测试集RMSE: {mlp_rmse:.4f}")
    
    # 7.3 优化MLP
    print("\n" + "=" * 60)
    print("3. 优化MLP（网格搜索超参数）")
    print("=" * 60)
    opt_mlp_filled = opt_mlp_imputer.impute(test_data_missing, test_hour)
    opt_mlp_rmse = calculate_rmse(test_data, opt_mlp_filled, test_missing_mask)
    results['优化MLP'] = {'rmse': opt_mlp_rmse, 'filled_data': opt_mlp_filled}
    print(f"  测试集RMSE: {opt_mlp_rmse:.4f}")
    
    # 保存优化结果信息
    if opt_mlp_imputer.best_params is not None:
        results['优化MLP']['best_params'] = opt_mlp_imputer.best_params
        print(f"  最优超参数: {opt_mlp_imputer.best_params}")
    
    # 6. 打印最终结果
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
    
    # 打印优化详情
    if '优化MLP' in results and 'best_params' in results['优化MLP']:
        print("\n" + "=" * 80)
        print("优化MLP 超参数详情")
        print("=" * 80)
        for param, value in results['优化MLP']['best_params'].items():
            print(f"  {param}: {value}")
    
    # 8. 绘制三种方法对比图
    print("\n" + "=" * 80)
    print("生成可视化对比图")
    print("=" * 80)
    plot_three_methods_comparison(test_data, test_missing_mask, results, target_col)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    import sys

    run_test()
