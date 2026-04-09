"""
EDA3 - 缺失值填充方法性能对比测试（简化版）
测试流程：
1. 截取日前价格最长连续片段，填充日前价格缺失值
2. 使用日前价格作为特征填充实时价格缺失值
3. 对比线性插值、XGBoost的填充效果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error
import warnings

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(data_path: str = None) -> pd.DataFrame:
    """加载数据"""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', '市场边界_出清价格总表.csv')
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['时间戳'].str.split('_').str[0])
    return df


def find_longest_segment(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """查找目标列最长连续非缺失片段"""
    valid_mask = data[target_col].notna().values
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        raise ValueError(f"列 '{target_col}' 没有有效值")
    
    segments = []
    start = valid_indices[0]
    prev = valid_indices[0]
    for idx in valid_indices[1:]:
        if idx != prev + 1:
            segments.append((start, prev + 1))
            start = idx
        prev = idx
    segments.append((start, prev + 1))
    
    longest = max(segments, key=lambda x: x[1] - x[0])
    return data.iloc[longest[0]:longest[1]].copy()


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取增强特征（不使用实时数据）"""
    features = pd.DataFrame(index=df.index)
    n = len(df)
    
    # 基础时间特征
    features['is_peak'] = df['是否高峰时段'].values if '是否高峰时段' in df.columns else ((features['hour'] >= 8) & (features['hour'] <= 15)).astype(int)
    features['day_of_week'] = df['星期'].values if '星期' in df.columns else np.zeros(n)
    features['month'] = df['月'].values if '月' in df.columns else np.ones(n)
    
    # 周期性编码（正弦/余弦变换）
    features['hour_sin'] = np.sin(2 * np.pi * df['小时'] / 24)
    features['dow_cos'] = np.cos(2 * np.pi * df['星期'] / 7)
    
    # 日前市场边界特征（不使用实时数据）
    dayahead_boundary_cols = [
        '系统负荷-日前',
        '风电出力-日前',
        '光伏出力-日前',
        '水电出力-日前',
        '联络线计划-日前',
    ]
    for col in dayahead_boundary_cols:
        if col in df.columns:
            features[col] = df[col].values
    
    return features


def generate_missing(data: np.ndarray, missing_rate: float = 0.2, 
                     min_gap: int = 24, max_gap: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """生成随机连续缺失值"""
    n = len(data)
    max_missing = int(n * missing_rate)
    data_missing = data.copy()
    missing_mask = np.zeros(n, dtype=bool)
    
    current_missing = 0
    while current_missing < max_missing:
        remaining = max_missing - current_missing
        if remaining < min_gap:
            break
        gap_len = np.random.randint(min_gap, min(max_gap, remaining) + 1)
        max_start = n - gap_len
        if max_start <= 0:
            break
        start = np.random.randint(0, max_start)
        end = start + gap_len
        
        if not np.any(missing_mask[start:end]):
            data_missing[start:end] = np.nan
            missing_mask[start:end] = True
            current_missing += gap_len
    
    return data_missing, missing_mask


class LinearImputer:
    """线性插值基线"""
    def __init__(self):
        self.name = "线性插值"
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        return pd.Series(data).interpolate(method='linear').values


class XGBImputer:
    """XGBoost填充器"""
    def __init__(self, window_size: int = 48, output_size: int = 24,
                 n_estimators: int = 100, max_depth: int = 6):
        self.name = "XGBoost"
        self.window_size = window_size
        self.output_size = output_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []
        self.feature_names = None
        self.n_features = 0
    
    def fit(self, data: np.ndarray, missing_mask: np.ndarray,
            features_df: pd.DataFrame = None, dayahead_filled: np.ndarray = None,
            original: np.ndarray = None):
        """训练XGBoost模型
        
        Parameters
        ----------
        data : np.ndarray
            含缺失值的数据
        missing_mask : np.ndarray
            缺失值掩码
        features_df : pd.DataFrame
            特征DataFrame（包含时间特征、市场边界特征等）
        dayahead_filled : np.ndarray
            填充后的日前价格（用于实时价格填充）
        original : np.ndarray
            原始完整数据（用于训练）
        """
        try:
            from models.tree_models import XGBoostModel, has_xgboost
            if not has_xgboost:
                print("  警告：XGBoost不可用")
                return
        except ImportError:
            print("  警告：无法导入XGBoost")
            return
        
        target = original if original is not None else pd.Series(data).interpolate().values
        input_data = pd.Series(data).interpolate().values
        input_data = np.nan_to_num(input_data, nan=np.nanmean(input_data))
        
        valid_idx = np.where(~missing_mask)[0]
        X_list = [[] for _ in range(self.output_size)]
        y_list = [[] for _ in range(self.output_size)]
        
        # 构建特征名称列表
        self.feature_names = [f'price_t-{self.window_size-i}' for i in range(self.window_size)]
        if features_df is not None:
            self.feature_names.extend(features_df.columns.tolist())
        else:
            self.feature_names.extend(['hour', 'is_peak'])
        if dayahead_filled is not None:
            self.feature_names.append('dayahead_price')
        
        for i in range(len(valid_idx) - self.window_size - self.output_size + 1):
            idx = valid_idx[i]
            end_idx = idx + self.window_size
            target_end = end_idx + self.output_size
            
            if target_end > len(data):
                continue
            if np.any(missing_mask[idx:end_idx]) or np.any(missing_mask[end_idx:target_end]):
                continue
            
            # 历史价格窗口
            price_hist = input_data[idx:end_idx]
            
            # 其他特征
            other_features = []
            if features_df is not None:
                other_features.extend(features_df.iloc[end_idx].values)
            else:
                hour = end_idx % 24
                peak = 1 if 8 <= hour <= 15 else 0
                other_features.extend([hour, peak])
            
            if dayahead_filled is not None:
                other_features.append(dayahead_filled[end_idx])
            
            X = np.concatenate([price_hist, other_features])
            
            for j in range(self.output_size):
                X_list[j].append(X)
                y_list[j].append(target[end_idx + j])
        
        if len(X_list[0]) < 10:
            print("  警告：训练样本不足")
            return
        
        self.n_features = len(X_list[0][0])
        print(f"  训练样本: {len(X_list[0])}, 输入维度: {self.n_features}")
        print(f"  特征列表: {self.feature_names}")
        
        self.models = []
        for j in range(self.output_size):
            X_train = np.array(X_list[j], dtype=np.float32)
            y_train = np.array(y_list[j], dtype=np.float32)
            
            if np.any(np.isnan(X_train)):
                X_train = np.nan_to_num(X_train, nan=0.0)
            if np.any(np.isnan(y_train)):
                y_train = np.nan_to_num(y_train, nan=np.nanmean(y_train))
            
            model = XGBoostModel(
                name=f"XGB_{j}",
                multi_output=False,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models.append(model)
        print(f"  XGBoost训练完成: {len(self.models)} 个模型")
    
    def get_feature_importance(self) -> np.ndarray:
        """获取平均特征重要性
        
        Returns
        -------
        importance : np.ndarray
            各特征的平均重要性
        """
        if len(self.models) == 0:
            return None
        
        importances = []
        for model in self.models:
            imp = model.get_feature_importance()
            if imp is not None:
                importances.append(imp)
        
        if len(importances) == 0:
            return None
        
        return np.mean(importances, axis=0)
    
    def impute(self, data: np.ndarray, features_df: pd.DataFrame = None,
               dayahead_filled: np.ndarray = None) -> np.ndarray:
        """填充缺失值
        
        Parameters
        ----------
        data : np.ndarray
            含缺失值的数据
        features_df : pd.DataFrame
            特征DataFrame
        dayahead_filled : np.ndarray
            填充后的日前价格
            
        Returns
        -------
        filled : np.ndarray
            填充后的数据
        """
        if len(self.models) == 0:
            return LinearImputer().impute(data)
        
        filled = data.copy()
        missing_mask = np.isnan(data)
        if not np.any(missing_mask):
            return filled
        
        gap_indices = np.where(missing_mask)[0]
        gap_groups = []
        current = [gap_indices[0]]
        for idx in gap_indices[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                gap_groups.append(current)
                current = [idx]
        gap_groups.append(current)
        
        for gap in gap_groups:
            pos = gap[0]
            end_pos = gap[-1] + 1
            
            while pos < end_pos:
                predict_len = min(self.output_size, end_pos - pos)
                
                if pos >= self.window_size:
                    window = filled[pos - self.window_size:pos]
                else:
                    window = np.pad(filled[:pos], (self.window_size - pos, 0), mode='edge')
                
                window = pd.Series(window).interpolate().values
                
                # 构建其他特征
                other_features = []
                if features_df is not None:
                    other_features.extend(features_df.iloc[pos].values)
                else:
                    hour = pos % 24
                    peak = 1 if 8 <= hour <= 15 else 0
                    other_features.extend([hour, peak])
                
                if dayahead_filled is not None:
                    other_features.append(dayahead_filled[pos])
                
                X = np.concatenate([window, other_features]).reshape(1, -1)
                
                preds = []
                for j in range(min(predict_len, len(self.models))):
                    preds.append(self.models[j].predict(X)[0])
                
                filled[pos:pos + len(preds)] = preds
                pos += len(preds)
        
        return filled


def calc_rmse(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算RMSE"""
    orig_vals = original[mask]
    filled_vals = filled[mask]
    filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    return np.sqrt(mean_squared_error(orig_vals, filled_vals))


def calc_mae(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算MAE（平均绝对误差）"""
    orig_vals = original[mask]
    filled_vals = filled[mask]
    filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    return np.mean(np.abs(orig_vals - filled_vals))


def calc_smape(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算sMAPE（对称平均绝对百分比误差）
    
    sMAPE = (1/n) * Σ(|预测值-实际值| / ((|预测值|+|实际值|)/2)) * 100%
    """
    orig_vals = original[mask]
    filled_vals = filled[mask]
    
    # 处理NaN值
    filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    
    # 避免除以0
    denominator = (np.abs(orig_vals) + np.abs(filled_vals)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape = np.mean(np.abs(orig_vals - filled_vals) / denominator) * 100
    return smape


def calc_metrics(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> dict:
    """计算所有评估指标
    
    Returns
    -------
    metrics : dict
        包含RMSE、MAE、sMAPE的字典
    """
    return {
        'rmse': calc_rmse(original, filled, mask),
        'mae': calc_mae(original, filled, mask),
        'smape': calc_smape(original, filled, mask)
    }


def plot_comparison(test_data: np.ndarray, test_mask: np.ndarray, 
                    results: Dict, title: str = "填充效果对比"):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    methods = ['线性插值', 'XGBoost']
    
    missing_idx = np.where(test_mask)[0]
    gap_groups = []
    if len(missing_idx) > 0:
        current = [missing_idx[0]]
        for idx in missing_idx[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                gap_groups.append((current[0], current[-1] + 1))
                current = [idx]
        gap_groups.append((current[0], current[-1] + 1))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        ax.plot(test_data, label='实际值', color='black', linewidth=1.5, alpha=0.8)
        
        if method in results:
            ax.plot(results[method]['filled'], label='填充值', 
                   color='red', linewidth=1.5, alpha=0.7, linestyle='--')
        
        for start, end in gap_groups:
            ax.axvspan(start, end, alpha=0.15, color='blue')
        if gap_groups:
            ax.axvspan(0, 0, alpha=0.15, color='blue', label='缺失区域')
        
        rmse = calc_rmse(test_data, results[method]['filled'], test_mask) if method in results else 0
        ax.set_title(f'{method} (RMSE: {rmse:.2f})', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(test_data))
    
    axes[-1].set_xlabel('时间点', fontsize=12)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), 'eda3_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存：{save_path}")
    plt.close(fig)


def print_results(results: Dict, target_name: str):
    """打印评估结果（包含RMSE、MAE、sMAPE）"""
    print(f"\n{'='*80}")
    print(f"{target_name}填充结果")
    print(f"{'='*80}")
    
    # 获取基线（线性插值）
    baseline_rmse = results['线性插值']['rmse']
    baseline_mae = results['线性插值']['mae']
    baseline_smape = results['线性插值']['smape']
    
    # 打印表头
    print(f"{'方法':<14} {'RMSE':>12} {'改进':>8} {'MAE':>12}  {'sMAPE(%)':>12}")
    print("-" * 80)
    
    # 按RMSE排序打印
    for name, res in sorted(results.items(), key=lambda x: x[1]['rmse']):
        rmse = res['rmse']
        mae = res['mae']
        smape = res['smape']
        
        rmse_improvement = (baseline_rmse - rmse) / baseline_rmse * 100
        
        print(f"{name:<14} {rmse:>12.2f} {rmse_improvement:>7.1f}% {mae:>12.2f}{smape:>12.2f}")


def plot_feature_importance(imputer: XGBImputer, title: str = "特征重要性", save_path: str = None):
    """绘制特征重要性图
    
    Parameters
    ----------
    imputer : XGBImputer
        XGBoost填充器实例
    title : str
        图表标题
    save_path : str
        保存路径
    """
    importance = imputer.get_feature_importance()
    if importance is None:
        print(f"  警告：无法获取{title}的特征重要性")
        return
    
    feature_names = imputer.feature_names
    
    # 创建DataFrame便于排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # 只显示重要性前20的特征
    importance_df = importance_df.tail(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), f'feature_importance_{title.replace(" ", "_")}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存：{save_path}")
    plt.close(fig)  # 关闭图形，避免显示问题
    
    # 打印前10重要特征
    print(f"\n{'='*60}")
    print(f"{title} - Top 10 重要特征")
    print(f"{'='*60}")
    top10 = importance_df.tail(10).iloc[::-1]
    for idx, row in top10.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")


class XGBImputerTuner:
    """XGBoost超参数调优器（网格搜索+交叉验证）"""
    
    def __init__(self, window_size: int = 6, output_size: int = 4, n_splits: int = 3):
        self.window_size = window_size
        self.output_size = output_size
        self.n_splits = n_splits
        self.best_params = None
        self.best_score = float('inf')
        
    def create_param_grid(self) -> list:
        """创建参数网格（方案2：固定max_depth=6，避免过拟合）"""
        param_grid = [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},   # 基准配置
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},   # 更多树
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1},   # 更多树
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05},  # 更低学习率
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},  # 更多树+低学习率
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05},  # 更多树+低学习率
        ]
        
        return param_grid
    
    def cross_validate(self, data: np.ndarray, missing_mask: np.ndarray,
                       features_df: pd.DataFrame, params: dict,
                       dayahead_filled: np.ndarray = None) -> float:
        """3折交叉验证"""
        n = len(data)
        fold_size = n // self.n_splits
        scores = []
        
        for fold in range(self.n_splits):
            # 划分训练集和验证集
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_splits - 1 else n
            
            train_indices = list(range(0, val_start)) + list(range(val_end, n))
            val_indices = list(range(val_start, val_end))
            
            if len(train_indices) < self.window_size + self.output_size + 10:
                continue
            
            # 准备数据
            train_data = data[train_indices]
            val_data = data[val_indices]
            train_missing = np.isnan(train_data)
            
            # 创建imputer并训练
            imputer = XGBImputer(
                window_size=self.window_size,
                output_size=self.output_size,
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth']
            )
            
            # 准备特征
            train_features = features_df.iloc[train_indices].reset_index(drop=True) if features_df is not None else None
            val_features = features_df.iloc[val_indices].reset_index(drop=True) if features_df is not None else None
            train_dayahead = dayahead_filled[train_indices] if dayahead_filled is not None else None
            
            try:
                imputer.fit(train_data, train_missing, train_features, train_dayahead, original=train_data)
                
                # 生成验证集缺失
                np.random.seed(42 + fold)
                val_missing, val_mask = generate_missing(val_data, missing_rate=0.2)
                
                # 填充
                val_dayahead = dayahead_filled[val_indices] if dayahead_filled is not None else None
                filled = imputer.impute(val_missing, val_features, val_dayahead)
                
                # 计算RMSE
                score = calc_rmse(val_data, filled, val_mask)
                scores.append(score)
            except Exception as e:
                print(f"  Fold {fold+1} 失败: {e}")
                scores.append(float('inf'))
        
        return np.mean(scores) if scores else float('inf')
    
    def tune(self, data: np.ndarray, features_df: pd.DataFrame = None,
             dayahead_filled: np.ndarray = None, verbose: bool = True) -> dict:
        """执行超参数调优"""
        param_grid = self.create_param_grid()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"开始超参数调优（网格搜索+{self.n_splits}折交叉验证）")
            print(f"{'='*70}")
            print(f"参数组合数: {len(param_grid)}")
        
        # 生成缺失值掩码（用于训练）
        np.random.seed(42)
        data_missing, missing_mask = generate_missing(data, missing_rate=0.2)
        
        best_params = None
        best_score = float('inf')
        
        for i, params in enumerate(param_grid):
            if verbose:
                print(f"\n[{i+1}/{len(param_grid)}] 测试参数: {params}")
            
            score = self.cross_validate(data, missing_mask, features_df, params, dayahead_filled)
            
            if verbose:
                print(f"  交叉验证RMSE: {score:.2f}")
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
                if verbose:
                    print(f"  ★ 新的最佳参数!")
        
        self.best_params = best_params
        self.best_score = best_score
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"超参数调优完成")
            print(f"{'='*70}")
            print(f"最佳参数: {best_params}")
            print(f"最佳交叉验证RMSE: {best_score:.2f}")
        
        return best_params


def run_pipeline(missing_rate: float = 0.2, tune_hyperparams: bool = True):
    """运行完整流程"""
    print("="*70)
    print("缺失值填充性能对比测试（增强特征工程版）")
    print("="*70)
    
    df = load_data()
    print(f"\n数据加载完成：{df.shape}")
    
    dayahead_col = [c for c in df.columns if '日前' in c and '价格' in c and '实时' not in c]
    realtime_col = [c for c in df.columns if '实时' in c and '价格' in c]
    
    if not dayahead_col or not realtime_col:
        print("错误：未找到价格列")
        return
    
    dayahead_col = dayahead_col[0]
    realtime_col = realtime_col[0]
    print(f"日前价格列：{dayahead_col}")
    print(f"实时价格列：{realtime_col}")
    
    # ==================== 阶段1：日前价格填充 ====================
    print("\n" + "="*70)
    print("阶段1：日前价格数据填充")
    print("="*70)
    
    df_dayahead = find_longest_segment(df, dayahead_col)
    print(f"日前价格最长连续段：{len(df_dayahead)}小时")
    
    # 提取增强特征
    features_df_da = extract_features(df_dayahead)
    print(f"特征数量：{len(features_df_da.columns)}")
    print(f"特征列表：{list(features_df_da.columns)}")
    
    dayahead_data = df_dayahead[dayahead_col].values.astype(float)
    
    n = len(dayahead_data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_data = dayahead_data[:train_end]
    test_data = dayahead_data[val_end:]
    test_features_da = features_df_da.iloc[val_end:].reset_index(drop=True)
    
    np.random.seed(42)
    train_missing, train_mask = generate_missing(train_data, missing_rate)
    
    # 超参数调优（阶段1）
    if tune_hyperparams:
        tuner_da = XGBImputerTuner(window_size=6, output_size=4, n_splits=3)
        best_params_da = tuner_da.tune(train_data, 
                                       features_df_da.iloc[:train_end].reset_index(drop=True),
                                       verbose=True)
        print(f"\n阶段1使用调优参数: {best_params_da}")
    else:
        best_params_da = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    
    xgb_da = XGBImputer(window_size=6, output_size=4, 
                        n_estimators=best_params_da['n_estimators'],
                        max_depth=best_params_da['max_depth'])
    xgb_da.fit(train_missing, train_mask, 
               features_df=features_df_da.iloc[:train_end].reset_index(drop=True),
               original=train_data)
    
    np.random.seed(123)
    test_missing, test_mask = generate_missing(test_data, missing_rate)
    
    results_da = {
        '线性插值': {'filled': LinearImputer().impute(test_missing), 'rmse': 0, 'mae': 0, 'smape': 0},
        'XGBoost': {'filled': xgb_da.impute(test_missing, test_features_da), 'rmse': 0, 'mae': 0, 'smape': 0}
    }
    
    for name in results_da:
        metrics = calc_metrics(test_data, results_da[name]['filled'], test_mask)
        results_da[name].update(metrics)
    
    print_results(results_da, "日前价格")
    
    # 保存填充后的日前价格
    dayahead_filled_full = np.zeros(len(dayahead_data))
    dayahead_filled_full[:train_end] = xgb_da.impute(
        np.where(np.isnan(train_missing), np.nan, train_data), 
        features_df_da.iloc[:train_end].reset_index(drop=True)
    )[:train_end]
    dayahead_filled_full[train_end:val_end] = LinearImputer().impute(dayahead_data[train_end:val_end])
    dayahead_filled_full[val_end:] = results_da['XGBoost']['filled']
    
    # ==================== 阶段2：实时价格填充 ====================
    print("\n" + "="*70)
    print("阶段2：实时价格数据填充（使用日前价格作为特征）")
    print("="*70)
    
    df_filled = df_dayahead.copy()
    df_filled.loc[:, '日前价格_填充'] = dayahead_filled_full
    
    # 提取增强特征
    features_df_rt = extract_features(df_filled)
    print(f"特征数量：{len(features_df_rt.columns)}")
    
    realtime_data = df_filled[realtime_col].values.astype(float)
    dayahead_filled_feat = df_filled['日前价格_填充'].values.astype(float)
    
    n2 = len(realtime_data)
    train_end2 = int(n2 * 0.6)
    val_end2 = int(n2 * 0.8)
    
    train_rt = realtime_data[:train_end2]
    test_rt = realtime_data[val_end2:]
    test_features_rt = features_df_rt.iloc[val_end2:].reset_index(drop=True)
    test_dayahead = dayahead_filled_feat[val_end2:]
    
    np.random.seed(42)
    train_rt_missing, train_rt_mask = generate_missing(train_rt, missing_rate)
    
    # 超参数调优（阶段2）
    if tune_hyperparams:
        tuner_rt = XGBImputerTuner(window_size=6, output_size=12, n_splits=3)
        best_params_rt = tuner_rt.tune(train_rt, 
                                       features_df_rt.iloc[:train_end2].reset_index(drop=True),
                                       dayahead_filled=dayahead_filled_feat[:train_end2],
                                       verbose=True)
        print(f"\n阶段2使用调优参数: {best_params_rt}")
    else:
        best_params_rt = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    
    xgb_rt = XGBImputer(window_size=6, output_size=12, 
                        n_estimators=best_params_rt['n_estimators'],
                        max_depth=best_params_rt['max_depth'])
    xgb_rt.fit(train_rt_missing, train_rt_mask,
               features_df=features_df_rt.iloc[:train_end2].reset_index(drop=True),
               dayahead_filled=dayahead_filled_feat[:train_end2],
               original=train_rt)
    
    np.random.seed(123)
    test_rt_missing, test_rt_mask = generate_missing(test_rt, missing_rate)
    
    results_rt = {
        '线性插值': {'filled': LinearImputer().impute(test_rt_missing), 'rmse': 0, 'mae': 0, 'smape': 0},
        'XGBoost': {'filled': xgb_rt.impute(test_rt_missing, test_features_rt, test_dayahead), 'rmse': 0, 'mae': 0, 'smape': 0}
    }
    
    for name in results_rt:
        metrics = calc_metrics(test_rt, results_rt[name]['filled'], test_rt_mask)
        results_rt[name].update(metrics)
    
    print_results(results_rt, "实时价格")
    
    print("\n" + "="*70)
    print("生成可视化")
    print("="*70)
    
    # 填充效果对比图
    plot_comparison(test_data, test_mask, results_da, "日前价格填充效果对比")
    plot_comparison(test_rt, test_rt_mask, results_rt, "实时价格填充效果对比（含日前价格特征）")
    
    # 特征重要性可视化
    plot_feature_importance(xgb_da, "阶段1-日前价格填充", 
                               os.path.join(os.path.dirname(__file__), 'feature_importance_dayahead.png'))
    plot_feature_importance(xgb_rt, "阶段2-实时价格填充", 
                               os.path.join(os.path.dirname(__file__), 'feature_importance_realtime.png'))
    
    return results_da, results_rt


def run_multi_missing_rate_experiment(missing_rates: list = None, tune_hyperparams: bool = True):
    """多缺失率实验
    
    Parameters
    ----------
    missing_rates : list
        缺失率列表，默认[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    tune_hyperparams : bool
        是否进行超参数调优
    """
    if missing_rates is None:
        missing_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("\n" + "="*80)
    print("多缺失率实验")
    print("="*80)
    print(f"缺失率列表: {missing_rates}")
    
    # 存储结果
    results = {
        'missing_rates': missing_rates,
        'dayahead': {'XGBoost': {'rmse': [], 'mae': []}, '线性插值': {'rmse': [], 'mae': []}},
        'realtime': {'XGBoost': {'rmse': [], 'mae': []}, '线性插值': {'rmse': [], 'mae': []}}
    }
    
    for i, mr in enumerate(missing_rates):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(missing_rates)}] 缺失率 = {mr*100:.0f}%")
        print(f"{'='*80}")
        
        # 运行单次实验
        results_da, results_rt = run_pipeline(missing_rate=mr, tune_hyperparams=tune_hyperparams)
        
        # 记录结果
        for method in ['XGBoost', '线性插值']:
            results['dayahead'][method]['rmse'].append(results_da[method]['rmse'])
            results['dayahead'][method]['mae'].append(results_da[method]['mae'])
            results['realtime'][method]['rmse'].append(results_rt[method]['rmse'])
            results['realtime'][method]['mae'].append(results_rt[method]['mae'])
    
    # 绘制综合评估大图
    plot_comprehensive_evaluation(results)
    
    return results


def plot_comprehensive_evaluation(results: dict, save_path: str = None):
    """绘制综合评估大图（4幅子图）
    
    Parameters
    ----------
    results : dict
        多缺失率实验结果
    save_path : str
        保存路径
    """
    missing_rates = results['missing_rates']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('缺失值填充性能综合评估', fontsize=16, fontweight='bold')
    
    # 颜色配置
    colors = {'XGBoost': '#2E86AB', '线性插值': '#A23B72'}
    markers = {'XGBoost': 'o', '线性插值': 's'}
    
    # 子图1: 日前价格 - RMSE
    ax1 = axes[0, 0]
    for method in ['XGBoost', '线性插值']:
        ax1.plot(missing_rates, results['dayahead'][method]['rmse'], 
                marker=markers[method], color=colors[method], 
                linewidth=2, markersize=8, label=method)
    ax1.set_xlabel('缺失率', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('日前价格 - RMSE', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(missing_rates)
    ax1.set_xticklabels([f'{mr*100:.0f}%' for mr in missing_rates])
    
    # 子图2: 日前价格 - MAE
    ax2 = axes[0, 1]
    for method in ['XGBoost', '线性插值']:
        ax2.plot(missing_rates, results['dayahead'][method]['mae'], 
                marker=markers[method], color=colors[method], 
                linewidth=2, markersize=8, label=method)
    ax2.set_xlabel('缺失率', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('日前价格 - MAE', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(missing_rates)
    ax2.set_xticklabels([f'{mr*100:.0f}%' for mr in missing_rates])
    
    # 子图3: 实时价格 - RMSE
    ax3 = axes[1, 0]
    for method in ['XGBoost', '线性插值']:
        ax3.plot(missing_rates, results['realtime'][method]['rmse'], 
                marker=markers[method], color=colors[method], 
                linewidth=2, markersize=8, label=method)
    ax3.set_xlabel('缺失率', fontsize=12)
    ax3.set_ylabel('RMSE', fontsize=12)
    ax3.set_title('实时价格 - RMSE', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(missing_rates)
    ax3.set_xticklabels([f'{mr*100:.0f}%' for mr in missing_rates])
    
    # 子图4: 实时价格 - MAE
    ax4 = axes[1, 1]
    for method in ['XGBoost', '线性插值']:
        ax4.plot(missing_rates, results['realtime'][method]['mae'], 
                marker=markers[method], color=colors[method], 
                linewidth=2, markersize=8, label=method)
    ax4.set_xlabel('缺失率', fontsize=12)
    ax4.set_ylabel('MAE', fontsize=12)
    ax4.set_title('实时价格 - MAE', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(missing_rates)
    ax4.set_xticklabels([f'{mr*100:.0f}%' for mr in missing_rates])
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'comprehensive_evaluation.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n综合评估图已保存：{save_path}")
    plt.close(fig)
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("多缺失率实验结果汇总")
    print("="*80)
    
    print("\n【日前价格】")
    print(f"{'缺失率':<10} {'XGBoost RMSE':<15} {'线性插值 RMSE':<15} {'XGBoost MAE':<15} {'线性插值 MAE':<15}")
    print("-"*70)
    for i, mr in enumerate(missing_rates):
        print(f"{mr*100:>6.0f}%    "
              f"{results['dayahead']['XGBoost']['rmse'][i]:<14.2f} "
              f"{results['dayahead']['线性插值']['rmse'][i]:<14.2f} "
              f"{results['dayahead']['XGBoost']['mae'][i]:<14.2f} "
              f"{results['dayahead']['线性插值']['mae'][i]:<14.2f}")
    
    print("\n【实时价格】")
    print(f"{'缺失率':<10} {'XGBoost RMSE':<15} {'线性插值 RMSE':<15} {'XGBoost MAE':<15} {'线性插值 MAE':<15}")
    print("-"*70)
    for i, mr in enumerate(missing_rates):
        print(f"{mr*100:>6.0f}%    "
              f"{results['realtime']['XGBoost']['rmse'][i]:<14.2f} "
              f"{results['realtime']['线性插值']['rmse'][i]:<14.2f} "
              f"{results['realtime']['XGBoost']['mae'][i]:<14.2f} "
              f"{results['realtime']['线性插值']['mae'][i]:<14.2f}")


if __name__ == "__main__":
    # 运行多缺失率实验
    run_multi_missing_rate_experiment(tune_hyperparams=True)
