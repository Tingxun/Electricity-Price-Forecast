"""
可视化工具模块
用于生成预测结果的可视化图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

logger = logging.getLogger(__name__)


class Visualizer:
    """
    可视化器类
    提供各种可视化功能
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        初始化可视化器
        
        Parameters
        ----------
        output_dir : str
            图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"可视化器初始化完成，输出目录: {output_dir}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str, dates: Optional[pd.DatetimeIndex] = None,
                        save: bool = True) -> str:
        """
        绘制预测值与真实值对比图
        
        Parameters
        ----------
        y_true : np.ndarray
            真实值，形状为 (n_samples, 24)
        y_pred : np.ndarray
            预测值，形状为 (n_samples, 24)
        model_name : str
            模型名称
        dates : pd.DatetimeIndex, optional
            日期索引
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径（如果save=True）
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} - 预测结果对比', fontsize=16, fontweight='bold')
        
        # 选择几个样本进行展示
        sample_indices = [0, len(y_true)//4, len(y_true)//2, 3*len(y_true)//4]
        sample_indices = [i for i in sample_indices if i < len(y_true)]
        
        for idx, (ax, sample_idx) in enumerate(zip(axes.flat, sample_indices)):
            hours = np.arange(24)
            
            ax.plot(hours, y_true[sample_idx], 'b-', label='真实值', linewidth=2, marker='o')
            ax.plot(hours, y_pred[sample_idx], 'r--', label='预测值', linewidth=2, marker='s')
            
            ax.set_xlabel('小时', fontsize=11)
            ax.set_ylabel('电价', fontsize=11)
            ax.set_title(f'样本 {sample_idx}', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(hours[::2])
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'{model_name}_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测对比图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str, save: bool = True) -> str:
        """
        绘制残差分析图
        
        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值
        model_name : str
            模型名称
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - 残差分析', fontsize=16, fontweight='bold')
        
        # 1. 残差分布直方图
        axes[0, 0].hist(residuals.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('残差', fontsize=11)
        axes[0, 0].set_ylabel('频数', fontsize=11)
        axes[0, 0].set_title('残差分布', fontsize=12)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差vs预测值散点图
        axes[0, 1].scatter(y_pred.flatten(), residuals.flatten(), alpha=0.5, s=10)
        axes[0, 1].set_xlabel('预测值', fontsize=11)
        axes[0, 1].set_ylabel('残差', fontsize=11)
        axes[0, 1].set_title('残差 vs 预测值', fontsize=12)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差vs真实值散点图
        axes[1, 0].scatter(y_true.flatten(), residuals.flatten(), alpha=0.5, s=10)
        axes[1, 0].set_xlabel('真实值', fontsize=11)
        axes[1, 0].set_ylabel('残差', fontsize=11)
        axes[1, 0].set_title('残差 vs 真实值', fontsize=12)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q图（检验正态性）
        from scipy import stats
        stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('残差Q-Q图', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'{model_name}_residuals.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"残差分析图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_feature_importance(self, importance: np.ndarray, 
                               feature_names: List[str],
                               model_name: str,
                               top_n: int = 20,
                               save: bool = True) -> str:
        """
        绘制特征重要性图
        
        Parameters
        ----------
        importance : np.ndarray
            特征重要性值
        feature_names : list
            特征名称列表
        model_name : str
            模型名称
        top_n : int
            显示前N个重要特征
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径
        """
        # 排序并选择Top N
        indices = np.argsort(importance)[::-1][:top_n]
        top_importance = importance[indices]
        top_features = [feature_names[i] for i in indices]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importance, align='center', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(f'{model_name} - 特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_model_comparison(self, results_df: pd.DataFrame, 
                             metrics: List[str] = ['mae', 'rmse', 'smape'],
                             save: bool = True) -> str:
        """
        绘制模型性能对比图
        
        Parameters
        ----------
        results_df : pd.DataFrame
            评估结果数据框
        metrics : list
            要对比的指标列表
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
        
        for idx, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
                
            ax = axes[idx]
            
            # 按指标排序
            sorted_df = results_df.sort_values(metric)
            
            # 绘制柱状图
            bars = ax.bar(range(len(sorted_df)), sorted_df[metric], 
                         color=colors, alpha=0.8, edgecolor='black')
            
            # 设置标签
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df['name'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(f'{metric.upper()} 对比', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, sorted_df[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"模型对比图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_hourly_performance(self, hourly_metrics: Dict[str, List[float]],
                               model_names: List[str],
                               metric_name: str = 'MAE',
                               save: bool = True) -> str:
        """
        绘制每小时性能对比图
        
        Parameters
        ----------
        hourly_metrics : dict
            每个模型每小时的指标值
        model_names : list
            模型名称列表
        metric_name : str
            指标名称
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        hours = np.arange(24)
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            if model_name in hourly_metrics:
                ax.plot(hours, hourly_metrics[model_name], 
                       marker='o', label=model_name, linewidth=2, 
                       color=colors[i], markersize=6)
        
        ax.set_xlabel('小时', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'每小时{metric_name}对比', fontsize=14, fontweight='bold')
        ax.set_xticks(hours)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'hourly_{metric_name.lower()}_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"每小时性能图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_time_series(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                        dates: pd.DatetimeIndex,
                        sample_days: int = 7,
                        save: bool = True) -> str:
        """
        绘制时间序列预测对比图
        
        Parameters
        ----------
        y_true : np.ndarray
            真实值
        predictions : dict
            各模型的预测值
        dates : pd.DatetimeIndex
            日期索引
        sample_days : int
            展示的天数
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径
        """
        # 选择前sample_days天
        n_hours = sample_days * 24
        n_hours = min(n_hours, len(y_true))
        
        y_true_sample = y_true[:n_hours].flatten()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 绘制真实值
        ax.plot(range(n_hours), y_true_sample, 'k-', label='真实值', 
               linewidth=2, alpha=0.8)
        
        # 绘制各模型预测值
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            y_pred_sample = y_pred[:n_hours].flatten()
            ax.plot(range(n_hours), y_pred_sample, '--', 
                   label=model_name, linewidth=1.5, alpha=0.7, color=colors[i])
        
        # 添加日期分隔线
        for day in range(1, sample_days):
            ax.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('时间（小时）', fontsize=12)
        ax.set_ylabel('电价', fontsize=12)
        ax.set_title(f'时间序列预测对比（前{sample_days}天）', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'time_series_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"时间序列对比图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def plot_heatmap(self, data: pd.DataFrame, 
                    title: str = '热力图',
                    save: bool = True) -> str:
        """
        绘制热力图
        
        Parameters
        ----------
        data : pd.DataFrame
            数据矩阵
        title : str
            图表标题
        save : bool
            是否保存图表
            
        Returns
        -------
        save_path : str
            保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': '值'}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"热力图已保存: {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def create_evaluation_dashboard(self, y_true: np.ndarray, 
                                   predictions: Dict[str, np.ndarray],
                                   results_df: pd.DataFrame,
                                   model_names: List[str]) -> List[str]:
        """
        创建完整的评估仪表板（生成所有图表）
        
        Parameters
        ----------
        y_true : np.ndarray
            真实值
        predictions : dict
            各模型的预测值
        results_df : pd.DataFrame
            评估结果数据框
        model_names : list
            模型名称列表
            
        Returns
        -------
        saved_paths : list
            所有保存的图表路径列表
        """
        saved_paths = []
        
        logger.info("开始生成评估仪表板...")
        
        # 1. 模型性能对比图
        path = self.plot_model_comparison(results_df)
        saved_paths.append(path)
        
        # 2. 每个模型的预测对比图
        for model_name in model_names[:3]:  # 只展示前3个模型
            if model_name in predictions:
                path = self.plot_predictions(y_true, predictions[model_name], model_name)
                saved_paths.append(path)
        
        # 3. 每个模型的残差分析图
        for model_name in model_names[:3]:
            if model_name in predictions:
                path = self.plot_residuals(y_true, predictions[model_name], model_name)
                saved_paths.append(path)
        
        # 4. 每小时MAE对比图
        if 'hourly_mae' in results_df.columns:
            hourly_mae = {}
            for _, row in results_df.iterrows():
                if row['name'] in model_names:
                    hourly_mae[row['name']] = row['hourly_mae']
            
            if hourly_mae:
                path = self.plot_hourly_performance(hourly_mae, model_names, 'MAE')
                saved_paths.append(path)
        
        # 5. 每小时sMAPE对比图
        if 'hourly_smape' in results_df.columns:
            hourly_smape = {}
            for _, row in results_df.iterrows():
                if row['name'] in model_names:
                    hourly_smape[row['name']] = row['hourly_smape']
            
            if hourly_smape:
                path = self.plot_hourly_performance(hourly_smape, model_names, 'sMAPE')
                saved_paths.append(path)
        
        # 6. 时间序列对比图
        if predictions:
            path = self.plot_time_series(y_true, predictions, 
                                        pd.date_range('2025-03-01', periods=len(y_true), freq='H'))
            saved_paths.append(path)
        
        logger.info(f"评估仪表板生成完成，共 {len(saved_paths)} 个图表")
        
        return saved_paths


def main():
    """
    测试可视化功能
    """
    # 创建测试数据
    np.random.seed(42)
    y_true = np.random.randn(100, 24) * 10 + 50
    y_pred = y_true + np.random.randn(100, 24) * 2
    
    # 创建可视化器
    visualizer = Visualizer(output_dir="test_figures")
    
    # 测试各种图表
    visualizer.plot_predictions(y_true, y_pred, "TestModel", save=True)
    visualizer.plot_residuals(y_true, y_pred, "TestModel", save=True)
    
    # 测试特征重要性
    importance = np.random.rand(20)
    feature_names = [f"Feature_{i}" for i in range(20)]
    visualizer.plot_feature_importance(importance, feature_names, "TestModel", save=True)
    
    # 测试模型对比
    results_df = pd.DataFrame({
        'name': ['Model1', 'Model2', 'Model3'],
        'mae': [5.2, 4.8, 5.5],
        'rmse': [7.1, 6.5, 7.8],
        'smape': [8.3, 7.7, 8.9]
    })
    visualizer.plot_model_comparison(results_df, save=True)
    
    print("可视化测试完成！")


if __name__ == '__main__':
    main()
