"""
缺失值填充方法性能对比测试
测试流程：
1. 截取实时价格最长连续片段
2. 对比线性插值、LightGBM的填充效果
n"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error
import warnings

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def load_data():
    """加载并合并市场边界数据和气象数据"""
    market_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'processed_市场总表.csv')
    weather_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'processed_气象总表.csv')
    
    # 加载市场边界数据
    market_df = pd.read_csv(market_data_path)
    market_df['datetime'] = pd.to_datetime(market_df['时间戳'].str.split('_').str[0])
    
    # 加载气象数据
    weather_df = pd.read_csv(weather_data_path)
    
    # 处理24:00时间格式问题
    def parse_datetime(date_str, time_str):
        if time_str == '24:00':
            dt = pd.to_datetime(date_str) + pd.Timedelta(days=1)
            return dt
        else:
            return pd.to_datetime(date_str + ' ' + time_str)
    
    weather_df['datetime'] = weather_df.apply(
        lambda row: parse_datetime(row['日期'], row['时段']), 
        axis=1
    )
    
    # 提取气象特征
    weather_features = extract_weather_features(weather_df)
    
    # 合并气象特征到气象数据框
    weather_df_with_features = weather_df[['datetime']].copy()
    weather_df_with_features = pd.concat([weather_df_with_features, weather_features], axis=1)
    
    # 合并市场数据和气象数据（按datetime）
    merged_df = market_df.merge(weather_df_with_features, on='datetime', how='left')
    
    print(f"市场边界数据：{market_df.shape}")
    print(f"气象数据：{weather_df.shape}")
    print(f"合并后数据：{merged_df.shape}")
    print(f"已添加气象特征：{len(weather_features.columns)} 个")
    
    return merged_df


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


def extract_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """提取气象特征"""
    features = pd.DataFrame(index=weather_df.index)
    
    # 获取所有列（排除日期、时段、datetime）
    weather_cols = [col for col in weather_df.columns if col not in ['日期', '时段', 'datetime']]
    
    # 按城市聚合气象特征
    cities = ['孝感市', '宜昌市', '武汉市', '荆州市', '荆门市', '襄阳市', '黄冈市', '随州市']
    
    for city in cities:
        city_cols = [col for col in weather_cols if col.startswith(city)]
        if not city_cols:
            continue
        
        # 对每种特征类型，保留所有数据源作为独立特征
        feature_types = ['温度', '风速', '总云量', '相对湿度', '压强', '辐照度', '降雨量']
        
        for feature_type in feature_types:
            type_cols = [col for col in city_cols if feature_type in col]
            for col in type_cols:
                # 保留原始列名作为特征名（简化名称）
                features[col] = weather_df[col].values
    
    return features


def extract_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取静态特征
    静态特征：预测哪一步直接使用哪一步对应的特征，在模型外进行处理
    """
    features = pd.DataFrame(index=df.index)
    
    # 基础时间特征
    features['is_peak'] = df['是否高峰时段']
    features['hour'] = df['小时']
    features['is_weekend'] = df['是否周末']
    features['month'] = df['月']
    
    # 周期性编码（正弦/余弦变换）
    features['hour_sin'] = np.sin(2 * np.pi * df['小时'] / 24)
    features['dow_cos'] = np.cos(2 * np.pi * df['星期'] / 7)
    
    # 市场边界特征
    dayahead_boundary_cols = [
        '系统负荷-实时',
        '风电出力-实时',
        '光伏出力-实时',
        '水电出力-实时',
        '联络线计划-实时',
    ]
    for col in dayahead_boundary_cols:
        if col in df.columns:
            features[col] = df[col].values
    
    # 添加气象特征（以下划线分隔的列名）
    cities = ['孝感市', '宜昌市', '武汉市', '荆州市', '荆门市', '襄阳市', '黄冈市', '恩施州', '十堰市', '黄石市', '咸宁市', '随州市']
    weather_cols = [col for col in df.columns if any(col.startswith(city.replace('-', '_')) for city in cities)]
    
    # for col in weather_cols:
    #     features[col] = df[col].values
    
    return features


def extract_dynamic_features(filled_data: np.ndarray, window_size: int, 
                             target_idx: int) -> np.ndarray:
    """
    提取动态特征（实时价格滞后特征 + 滚动特征 + 差分特征）
    动态特征：预测T步时使用T-1到T-window_size的历史滞后，历史滞后随着填充轮次变化
    
    包含的特征：
    1. 原始滞后特征: [price_t-window_size, ..., price_t-1]
    2. 滚动统计特征: mean, std, min, max, median
    3. 差分特征: 一阶差分、二阶差分
    
    Args:
        filled_data: 已填充的数据（包含已填充的部分）
        window_size: 滞后窗口大小
        target_idx: 当前预测目标的时间索引
        
    Returns:
        动态特征数组（包含滞后、滚动统计、差分特征）
    """
    if target_idx >= window_size:
        # 正常情况：有足够的历史数据
        window = filled_data[target_idx - window_size:target_idx]
    else:
        # 边界情况：历史数据不足，使用边缘填充
        window = np.pad(filled_data[:target_idx], (window_size - target_idx, 0), mode='edge')
    
    # 对窗口中的缺失值进行线性插值（以防万一）
    window = pd.Series(window).interpolate().values
    
    features = []
    
    # 1. 原始滞后特征
    features.extend(window)
    
    # 2. 滚动统计特征（使用不同窗口大小）
    window_series = pd.Series(window)
    
    # 短窗口滚动特征（最近6小时）
    short_window = min(6, window_size)
    short_data = window[-short_window:]
    features.append(np.mean(short_data))  # 短窗口均值
    features.append(np.std(short_data) if len(short_data) > 1 else 0)  # 短窗口标准差
    features.append(np.min(short_data))  # 短窗口最小值
    features.append(np.max(short_data))  # 短窗口最大值
    
    # 中窗口滚动特征（最近12小时）
    mid_window = min(12, window_size)
    mid_data = window[-mid_window:]
    features.append(np.mean(mid_data))  # 中窗口均值
    features.append(np.std(mid_data) if len(mid_data) > 1 else 0)  # 中窗口标准差
    
    # 长窗口滚动特征（整个窗口）
    features.append(np.mean(window))  # 长窗口均值
    features.append(np.std(window) if len(window) > 1 else 0)  # 长窗口标准差
    features.append(np.median(window))  # 长窗口中位数
    
    # 3. 差分特征
    # 一阶差分（最近值与窗口均值的差）
    first_diff = window[-1] - window[-2] if len(window) >= 2 else 0
    features.append(first_diff)
    
    # 与窗口均值的差
    mean_diff = window[-1] - np.mean(window)
    features.append(mean_diff)
    
    # 与窗口最大/最小值的差
    max_diff = window[-1] - np.max(window)
    min_diff = window[-1] - np.min(window)
    features.append(max_diff)
    features.append(min_diff)
    
    # 4. 趋势特征（线性回归斜率）
    if len(window) >= 3:
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0] if np.std(window) > 0 else 0
        features.append(slope)
    else:
        features.append(0)
    
    return np.array(features, dtype=np.float32)

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


class FeaturePreprocessor:
    """特征预处理器：线性插值填充缺失值"""
    
    def __init__(self):
        self.feature_cols = None
    
    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换训练集特征"""
        df = features_df.copy()
        self.feature_cols = df.columns.tolist()
        
        # 线性插值填充缺失值
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """转换测试集特征"""
        df = features_df.copy()
        
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
        
        return df


class LightGBMImputerMultiOutput:
    """LightGBM多输出填充器"""
    def __init__(self, window_size: int = 48, output_size: int = 24,
                 n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, num_leaves: int = 31, 
                 feature_fraction: float = 0.8, bagging_fraction: float = 0.8, 
                 bagging_freq: int = 5):
        self.name = "LightGBM"
        self.window_size = window_size
        self.output_size = output_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.model = None
        self.feature_names = None
        self.n_features = 0
        self.static_feature_cols = None  # 存储静态特征列名
    
    def _build_feature_vector(self, filled_data: np.ndarray, 
                              static_features_df: pd.DataFrame = None,
                              target_start_idx: int = 0) -> np.ndarray:
        """
        构建特征向量 - 统一训练和预测的特征工程流程
        
        特征组成：
        1. 动态特征：使用T-1到T-window_size的历史滞后
        2. 静态特征：预测哪一步直接使用哪一步对应的特征
        
        Args:
            filled_data: 已填充的价格数据
            static_features_df: 静态特征DataFrame
            target_start_idx: 预测目标的起始索引
            
        Returns:
            特征向量
        """
        # 提取动态特征
        dynamic_features = extract_dynamic_features(
            filled_data, self.window_size, target_start_idx
        )
        
        # 提取静态特征
        static_features = []
        if static_features_df is not None:
            for j in range(self.output_size):
                feat_pos = min(target_start_idx + j, len(static_features_df) - 1)
                static_features.extend(static_features_df.iloc[feat_pos].values)
        else:
            # 回退：使用时间特征
            for j in range(self.output_size):
                hour = (target_start_idx + j) % 24
                peak = 1 if 8 <= hour <= 15 else 0
                static_features.extend([hour, peak])
        
        # 合并动态特征和静态特征
        feature_vector = np.concatenate([dynamic_features, static_features])
        
        return feature_vector
    
    def _build_feature_names(self, static_features_df: pd.DataFrame = None):
        """构建特征名称列表"""
        feature_names = []
        
        # 动态特征名称
        # 1. 原始滞后特征
        for i in range(self.window_size):
            feature_names.append(f'price_lag_{self.window_size - i}')
        
        # 2. 滚动统计特征（短窗口6小时）
        feature_names.extend([
            'rolling_mean_6h', 'rolling_std_6h', 'rolling_min_6h', 'rolling_max_6h'
        ])
        
        # 3. 滚动统计特征（中窗口12小时）
        feature_names.extend([
            'rolling_mean_12h', 'rolling_std_12h'
        ])
        
        # 4. 滚动统计特征（长窗口）
        feature_names.extend([
            'rolling_mean_full', 'rolling_std_full', 'rolling_median_full'
        ])
        
        # 5. 差分特征
        feature_names.extend([
            'diff_1st',           # 一阶差分
            'diff_from_mean',     # 与均值差
            'diff_from_max',      # 与最大值差
            'diff_from_min',      # 与最小值差
            'trend_slope'         # 趋势斜率
        ])
        
        # 静态特征名称
        if static_features_df is not None:
            static_cols = static_features_df.columns.tolist()
            for j in range(self.output_size):
                for col in static_cols:
                    feature_names.append(f'{col}_t+{j}')
            self.static_feature_cols = static_cols
        else:
            for j in range(self.output_size):
                feature_names.extend([f'hour_t+{j}', f'is_peak_t+{j}'])
        
        self.feature_names = feature_names
    
    def fit(self, data: np.ndarray, missing_mask: np.ndarray,
            features_df: pd.DataFrame = None, original: np.ndarray = None):
        """训练多输出LightGBM模型"""
        try:
            import lightgbm as lgb
            from sklearn.multioutput import MultiOutputRegressor
        except ImportError:
            print("  警告：无法导入LightGBM或sklearn，请安装：pip install lightgbm scikit-learn")
            return

        # 优先使用原始完整数据来生成训练样本
        if original is not None:
            input_data = original.copy()
            target = original.copy()
            use_original = True
        else:
            input_data = pd.Series(data).interpolate().values
            target = data.copy()
            use_original = False

        X_list = []
        y_list = []

        # 构建特征名称
        self._build_feature_names(features_df)

        # 构建训练样本
        n_samples = len(data)
        skipped = 0
        for idx in range(n_samples - self.window_size - self.output_size + 1):
            end_idx = idx + self.window_size
            target_end = end_idx + self.output_size

            if not use_original:
                if np.any(missing_mask[idx:end_idx]) or np.any(missing_mask[end_idx:target_end]):
                    skipped += 1
                    continue

            # 使用统一的特征工程方法构建特征向量
            X = self._build_feature_vector(input_data, features_df, end_idx)
            X_list.append(X)
            y_list.append(target[end_idx:target_end])

        if len(X_list) < 10:
            print(f"  警告：训练样本不足（仅{len(X_list)}个）")
            return

        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.float32)
        self.n_features = X_train.shape[1]

        print(f"  LightGBM训练样本: {len(X_list)}, 输入维度: {self.n_features}, 输出维度: {self.output_size}")
        if skipped > 0:
            print(f"  因缺失值跳过的样本: {skipped}")
        
        # 使用MultiOutputRegressor包装LightGBM
        base_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            objective='regression',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        self.model.fit(X_train, y_train)
        print(f"  LightGBM模型训练完成")
    
    def get_feature_importance(self) -> np.ndarray:
        """获取平均特征重要性"""
        if self.model is None:
            return None
        
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        return np.mean(importances, axis=0)
    
    def impute(self, data: np.ndarray, features_df: pd.DataFrame = None) -> np.ndarray:
        """填充缺失值"""
        if self.model is None:
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
                
                # 使用统一的特征工程方法构建特征向量
                X = self._build_feature_vector(filled, features_df, pos)
                X = X.reshape(1, -1)
                
                # 多输出预测
                preds = self.model.predict(X)[0][:predict_len]
                filled[pos:pos + predict_len] = preds
                pos += predict_len
        
        return filled


def calc_rmse(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算RMSE"""
    orig_vals = original[mask]
    filled_vals = filled[mask]
    filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    return np.sqrt(mean_squared_error(orig_vals, filled_vals))


def calc_mae(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算MAE"""
    orig_vals = original[mask]
    filled_vals = filled[mask]
    filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    return np.mean(np.abs(orig_vals - filled_vals))


def calc_smape(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """计算sMAPE"""
    orig_vals = original[mask]
    filled_vals = filled[mask]
    filled_vals = np.nan_to_num(filled_vals, nan=np.nanmean(filled_vals))
    
    denominator = (np.abs(orig_vals) + np.abs(filled_vals)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape = np.mean(np.abs(orig_vals - filled_vals) / denominator) * 100
    return smape


def calc_metrics(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> dict:
    """计算所有评估指标"""
    return {
        'rmse': calc_rmse(original, filled, mask),
        'mae': calc_mae(original, filled, mask),
        'smape': calc_smape(original, filled, mask)
    }


def plot_comparison(test_data: np.ndarray, test_mask: np.ndarray, 
                    results: Dict, title: str = "填充效果对比"):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    methods = ['线性插值', 'LightGBM']
    
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
        
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('电价', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), 'imputation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存：{save_path}")
    plt.close(fig)


def print_results(results: Dict, target_name: str):
    """打印评估结果"""
    print(f"\n{'='*80}")
    print(f"{target_name}填充结果")
    print(f"{'='*80}")
    
    # 获取基线（线性插值）
    baseline_rmse = results['线性插值']['rmse']
    baseline_mae = results['线性插值']['mae']
    baseline_smape = results['线性插值']['smape']
    
    # 打印表头
    print(f"{'方法':<20} {'RMSE':>12} {'改进':>8} {'MAE':>12}  {'sMAPE(%)':>12}")
    print("-" * 80)
    
    # 按RMSE排序打印
    for name, res in sorted(results.items(), key=lambda x: x[1]['rmse']):
        rmse = res['rmse']
        mae = res['mae']
        smape = res['smape']
        
        rmse_improvement = (baseline_rmse - rmse) / baseline_rmse * 100
        
        print(f"{name:<20} {rmse:>12.2f} {rmse_improvement:>7.1f}% {mae:>12.2f}{smape:>12.2f}")


def plot_feature_importance(imputer: LightGBMImputerMultiOutput, title: str = "特征重要性", 
                            save_path: str = None):
    """绘制特征重要性图"""
    importance = imputer.get_feature_importance()
    if importance is None:
        print(f"  警告：无法获取{title}的特征重要性")
        return
    
    feature_names = imputer.feature_names
    
    # 创建DataFrame便于排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # 合并同一基础特征的不同时间步（如 _t+0, _t+1 等）
    def extract_base_feature(feature_name):
        """提取基础特征名，去除时间步后缀"""
        if '_t+' in feature_name:
            return feature_name.rsplit('_t+', 1)[0]
        return feature_name
    
    importance_df['base_feature'] = importance_df['feature'].apply(extract_base_feature)
    
    # 按基础特征分组，计算平均重要性
    merged_importance = importance_df.groupby('base_feature')['importance'].mean().reset_index()
    merged_importance = merged_importance.sort_values('importance', ascending=True)
    
    # 只显示重要性前40的特征
    merged_importance = merged_importance.tail(40)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(merged_importance)))
    bars = ax.barh(merged_importance['base_feature'], merged_importance['importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'{title} - Top 40 Feature Importance (Merged)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存：{save_path}")
    plt.close(fig)


def run_pipeline(missing_rate: float = 0.2):
    """运行完整流程"""
    print("="*70)
    print("缺失值填充性能对比测试")
    print("="*70)
    
    # 加载已合并的数据
    df = load_data()
    
    # 只使用实时价格列
    realtime_col = [c for c in df.columns if '实时' in c and '价格' in c]
    
    if not realtime_col:
        print("错误：未找到实时价格列")
        return None
    
    realtime_col = realtime_col[0]
    print(f"实时价格列：{realtime_col}")
    
    # 查找实时价格最长连续段
    df_realtime = find_longest_segment(df, realtime_col)
    print(f"实时价格最长连续段：{len(df_realtime)}小时")
    
    # 提取特征
    features_df = extract_static_features(df_realtime)
    print(f"特征数量：{len(features_df.columns)}")
    print(f"特征列表：{list(features_df.columns)}")
    
    realtime_data = df_realtime[realtime_col].values.astype(float)

    n = len(realtime_data)
    train_end = int(n * 0.8)

    train_data = realtime_data[:train_end]
    test_data = realtime_data[train_end:]

    # 划分训练集和测试集特征
    train_features_raw = features_df.iloc[:train_end].reset_index(drop=True)
    test_features_raw = features_df.iloc[train_end:].reset_index(drop=True)
    
    # 特征预处理
    preprocessor = FeaturePreprocessor()
    train_features = preprocessor.fit_transform(train_features_raw)
    test_features = preprocessor.transform(test_features_raw)
    print(f"训练集特征: {train_features.shape}, 测试集特征: {test_features.shape}")
    
    np.random.seed(42)
    train_missing, train_mask = generate_missing(train_data, missing_rate)

    # 训练LightGBM填充器
    print("\n  训练LightGBM填充器...")
    lgb = LightGBMImputerMultiOutput(
        window_size=6,
        output_size=12,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=3,
        num_leaves=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5
    )
    lgb.fit(train_missing, train_mask,
                 features_df=train_features,
                 original=train_data)

    # 测试
    np.random.seed(123)
    test_missing, test_mask = generate_missing(test_data, missing_rate)

    # 对比两种方法
    results = {
        '线性插值': {'filled': LinearImputer().impute(test_missing), 'rmse': 0, 'mae': 0, 'smape': 0},
        'LightGBM': {'filled': lgb.impute(test_missing, test_features), 'rmse': 0, 'mae': 0, 'smape': 0}
    }
    
    for name in results:
        metrics = calc_metrics(test_data, results[name]['filled'], test_mask)
        results[name].update(metrics)
    
    print_results(results, "实时价格")
    
    print("\n" + "="*70)
    print("生成可视化")
    print("="*70)
    
    # 填充效果对比图
    plot_comparison(test_data, test_mask, results, "实时价格填充效果对比")
    
    # 特征重要性可视化
    plot_feature_importance(lgb, "实时价格填充")
    
    return results


def run_multi_missing_rate_experiment(missing_rates: list = None):
    """多缺失率实验"""
    if missing_rates is None:
        missing_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    print("\n" + "="*80)
    print("多缺失率实验（LightGBM）")
    print("="*80)
    print(f"缺失率列表: {missing_rates}")
    
    # 存储结果
    results = {
        'missing_rates': missing_rates,
        'realtime': {'LightGBM': {'rmse': [], 'mae': []}, '线性插值': {'rmse': [], 'mae': []}}
    }
    
    for i, mr in enumerate(missing_rates):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(missing_rates)}] 缺失率 = {mr*100:.0f}%")
        print(f"{'='*80}")
        
        # 运行单次实验
        results_rt = run_pipeline(missing_rate=mr)
        
        if results_rt is None:
            print("  警告：实验失败，跳过")
            continue
        
        # 记录结果
        for method in ['LightGBM', '线性插值']:
            results['realtime'][method]['rmse'].append(results_rt[method]['rmse'])
            results['realtime'][method]['mae'].append(results_rt[method]['mae'])
    
    # 绘制综合评估大图
    plot_comprehensive_evaluation(results)
    
    return results


def plot_comprehensive_evaluation(results: dict, save_path: str = None):
    """绘制综合评估大图"""
    missing_rates = results['missing_rates']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('实时电价缺失值填充性能综合评估', fontsize=16, fontweight='bold')
    
    # 颜色配置
    colors = {'LightGBM': '#2E86AB', '线性插值': '#A23B72'}
    markers = {'LightGBM': 'o', '线性插值': 's'}
    
    # 子图1: RMSE
    ax1 = axes[0]
    for method in ['LightGBM', '线性插值']:
        ax1.plot(missing_rates, results['realtime'][method]['rmse'], 
                marker=markers[method], color=colors[method], 
                linewidth=2, markersize=8, label=method)
    ax1.set_xlabel('缺失率', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('实时电价 - RMSE', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(missing_rates)
    ax1.set_xticklabels([f'{mr*100:.0f}%' for mr in missing_rates])
    
    # 子图2: MAE
    ax2 = axes[1]
    for method in ['LightGBM', '线性插值']:
        ax2.plot(missing_rates, results['realtime'][method]['mae'], 
                marker=markers[method], color=colors[method], 
                linewidth=2, markersize=8, label=method)
    ax2.set_xlabel('缺失率', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('实时电价 - MAE', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(missing_rates)
    ax2.set_xticklabels([f'{mr*100:.0f}%' for mr in missing_rates])
    
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
    
    print("\n【实时电价】")
    print(f"{'缺失率':<10} {'LightGBM RMSE':<20} {'线性插值 RMSE':<15} {'LightGBM MAE':<20} {'线性插值 MAE':<15}")
    print("-"*80)
    for i, mr in enumerate(missing_rates):
        print(f"{mr*100:>6.0f}%    "
              f"{results['realtime']['LightGBM']['rmse'][i]:<19.2f} "
              f"{results['realtime']['线性插值']['rmse'][i]:<14.2f} "
              f"{results['realtime']['LightGBM']['mae'][i]:<19.2f} "
              f"{results['realtime']['线性插值']['mae'][i]:<14.2f}")


if __name__ == "__main__":
    # 运行单次实验
    run_pipeline(missing_rate=0.4)
    
    # 运行多缺失率实验
    # run_multi_missing_rate_experiment()
