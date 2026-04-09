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


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """提取时间特征"""
    hour = df['小时'].values if '小时' in df.columns else np.arange(len(df)) % 24
    peak = df['是否高峰时段'].values if '是否高峰时段' in df.columns else ((hour >= 8) & (hour <= 15)).astype(int)
    return hour, peak


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
    
    def fit(self, data: np.ndarray, missing_mask: np.ndarray,
            hour_feat: np.ndarray = None, peak_feat: np.ndarray = None,
            dayahead_filled: np.ndarray = None, original: np.ndarray = None):
        """训练XGBoost模型"""
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
        
        for i in range(len(valid_idx) - self.window_size - self.output_size + 1):
            idx = valid_idx[i]
            end_idx = idx + self.window_size
            target_end = end_idx + self.output_size
            
            if target_end > len(data):
                continue
            if np.any(missing_mask[idx:end_idx]) or np.any(missing_mask[end_idx:target_end]):
                continue
            
            price_hist = input_data[idx:end_idx]
            hour = hour_feat[end_idx] if hour_feat is not None else end_idx % 24
            peak = peak_feat[end_idx] if peak_feat is not None else 0
            
            features = [hour, peak]
            if dayahead_filled is not None:
                features.append(dayahead_filled[end_idx])
            
            X = np.concatenate([price_hist, features])
            
            for j in range(self.output_size):
                X_list[j].append(X)
                y_list[j].append(target[end_idx + j])
        
        if len(X_list[0]) < 10:
            print("  警告：训练样本不足")
            return
        
        print(f"  训练样本: {len(X_list[0])}, 输入维度: {len(X_list[0][0])}")
        
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
    
    def impute(self, data: np.ndarray, hour_feat: np.ndarray = None,
               peak_feat: np.ndarray = None, dayahead_filled: np.ndarray = None) -> np.ndarray:
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
                hour = hour_feat[pos] if hour_feat is not None else pos % 24
                peak = peak_feat[pos] if peak_feat is not None else 0
                
                features = [hour, peak]
                if dayahead_filled is not None:
                    features.append(dayahead_filled[pos])
                
                X = np.concatenate([window, features]).reshape(1, -1)
                
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
    plt.show()


def print_results(results: Dict, target_name: str):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"{target_name}填充结果")
    print(f"{'='*60}")
    
    baseline = results['线性插值']['rmse']
    print(f"{'方法':<20} {'RMSE':>12} {'改进':>12}")
    print("-" * 50)
    
    for name, res in sorted(results.items(), key=lambda x: x[1]['rmse']):
        rmse = res['rmse']
        improvement = (baseline - rmse) / baseline * 100
        print(f"{name:<20} {rmse:>12.2f} {improvement:>11.1f}%")


def run_pipeline(missing_rate: float = 0.05):
    """运行完整流程"""
    print("="*70)
    print("缺失值填充性能对比测试")
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
    
    hour_feat, peak_feat = extract_features(df_dayahead)
    dayahead_data = df_dayahead[dayahead_col].values.astype(float)
    
    n = len(dayahead_data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_data = dayahead_data[:train_end]
    test_data = dayahead_data[val_end:]
    test_hour = hour_feat[val_end:]
    test_peak = peak_feat[val_end:]
    
    np.random.seed(42)
    train_missing, train_mask = generate_missing(train_data, missing_rate)
    
    xgb_da = XGBImputer(window_size=48, output_size=24, n_estimators=100, max_depth=6)
    xgb_da.fit(train_missing, train_mask, hour_feat[:train_end], peak_feat[:train_end], 
               original=train_data)
    
    np.random.seed(123)
    test_missing, test_mask = generate_missing(test_data, missing_rate)
    
    results_da = {
        '线性插值': {'filled': LinearImputer().impute(test_missing), 'rmse': 0},
        'XGBoost': {'filled': xgb_da.impute(test_missing, test_hour, test_peak), 'rmse': 0}
    }
    
    for name in results_da:
        results_da[name]['rmse'] = calc_rmse(test_data, results_da[name]['filled'], test_mask)
    
    print_results(results_da, "日前价格")
    
    # 保存填充后的日前价格
    dayahead_filled_full = np.zeros(len(dayahead_data))
    dayahead_filled_full[:train_end] = xgb_da.impute(
        np.where(np.isnan(train_missing), np.nan, train_data), 
        hour_feat[:train_end], peak_feat[:train_end]
    )[:train_end]
    dayahead_filled_full[train_end:val_end] = LinearImputer().impute(dayahead_data[train_end:val_end])
    dayahead_filled_full[val_end:] = results_da['XGBoost']['filled']
    
    # ==================== 阶段2：实时价格填充 ====================
    print("\n" + "="*70)
    print("阶段2：实时价格数据填充（使用日前价格作为特征）")
    print("="*70)
    
    df_filled = df_dayahead.copy()
    df_filled.loc[:, '日前价格_填充'] = dayahead_filled_full
    
    hour_feat2, peak_feat2 = extract_features(df_filled)
    realtime_data = df_filled[realtime_col].values.astype(float)
    dayahead_filled_feat = df_filled['日前价格_填充'].values.astype(float)
    
    n2 = len(realtime_data)
    train_end2 = int(n2 * 0.6)
    val_end2 = int(n2 * 0.8)
    
    train_rt = realtime_data[:train_end2]
    test_rt = realtime_data[val_end2:]
    test_hour2 = hour_feat2[val_end2:]
    test_peak2 = peak_feat2[val_end2:]
    test_dayahead = dayahead_filled_feat[val_end2:]
    
    np.random.seed(42)
    train_rt_missing, train_rt_mask = generate_missing(train_rt, missing_rate)
    
    xgb_rt = XGBImputer(window_size=48, output_size=24, n_estimators=100, max_depth=6)
    xgb_rt.fit(train_rt_missing, train_rt_mask,
               hour_feat2[:train_end2], peak_feat2[:train_end2],
               dayahead_filled=dayahead_filled_feat[:train_end2],
               original=train_rt)
    
    np.random.seed(123)
    test_rt_missing, test_rt_mask = generate_missing(test_rt, missing_rate)
    
    results_rt = {
        '线性插值': {'filled': LinearImputer().impute(test_rt_missing), 'rmse': 0},
        'XGBoost': {'filled': xgb_rt.impute(test_rt_missing, test_hour2, test_peak2, test_dayahead), 'rmse': 0}
    }
    
    for name in results_rt:
        results_rt[name]['rmse'] = calc_rmse(test_rt, results_rt[name]['filled'], test_rt_mask)
    
    print_results(results_rt, "实时价格")
    
    print("\n" + "="*70)
    print("生成可视化")
    print("="*70)
    plot_comparison(test_data, test_mask, results_da, "日前价格填充效果对比")
    plot_comparison(test_rt, test_rt_mask, results_rt, "实时价格填充效果对比（含日前价格特征）")
    
    print("\n测试完成！")


if __name__ == "__main__":
    run_pipeline()
