# 湖北省日前电价预测系统

基于多源异构数据的电力现货市场日前电价预测系统，支持多种机器学习模型的训练、评估和预测。

## 项目概述

本项目针对湖北省电力现货市场，利用市场边界信息（负荷预测、新能源出力、联络线计划等）和出清价格数据，构建日前电价预测模型。支持24小时逐点价格预测，为市场参与者提供决策支持。

## 项目结构

```
Electricity Price Forecast/
├── code/                       # 核心代码目录
│   ├── main.py                 # 主程序入口
│   ├── config.py               # 项目配置
│   ├── data_preprocessing.py   # 数据预处理
│   ├── generate_features.py    # 特征工程
│   ├── feature_engineering.py  # 特征工程类
│   ├── train.py                # 模型训练
│   ├── evaluate.py             # 模型评估
│   ├── predict.py              # 预测功能
│   ├── hyperparameter_tuning.py # 超参数调优
│   ├── data_loader.py          # 数据加载
│   └── ...
├── models/                     # 模型实现
│   ├── linear_models.py        # 线性模型（Lasso/Ridge等）
│   ├── tree_models.py          # 树模型（XGBoost/RF等）
│   ├── neural_networks.py      # 神经网络（LSTM/GRU等）
│   └── base_model.py           # 模型基类
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   ├── processed/              # 预处理后数据
│   └── features/               # 特征工程数据
├── utils/                      # 工具函数
│   ├── metrics.py              # 评估指标
│   └── visualization.py        # 可视化
├── saved_models/               # 保存的模型
├── results/                    # 结果输出
│   ├── predictions/            # 预测结果
│   ├── figures/                # 图表
│   └── logs/                   # 日志
└── notebooks/                  # Jupyter笔记本
```

## 环境要求

- Python 3.8+
- 主要依赖包：
  - pandas, numpy
  - scikit-learn
  - torch (神经网络模型)
  - xgboost (可选)
  - epftoolbox (可选，LEAR模型)

### 安装依赖

```bash
pip install pandas numpy scikit-learn torch
pip install xgboost  # 可选
pip install epftoolbox  # 可选
```

## 快速开始

### 1. 数据准备

将原始数据文件 `市场边界_出清价格总表.csv` 放入 `data/raw/` 目录。

数据应包含以下字段：
- 时间相关：日期、时段、小时、星期、月份等
- 市场边界：系统负荷（日前/实时）、风电/光伏/水电出力、联络线计划等
- 价格数据：平均出清价格（日前/实时）

### 2. 数据预处理

清洗原始数据，处理缺失值和异常值：

```bash
cd code/
python main.py preprocess
```

处理后的数据保存在 `data/processed/processed_data.csv`

### 3. 特征工程

从清洗后的数据构建特征：

```bash
python main.py features
```

生成的特征保存在 `data/features/features.csv`

### 4. 模型训练

训练指定模型：

```bash
# 训练单个模型
python main.py train --models Lasso

# 训练多个模型
python main.py train --models LinearRegression Ridge Lasso

# 训练所有模型
python main.py train
```

支持的模型：

| 类别 | 模型名称 |
|------|----------|
| 线性模型 | LinearRegression, Ridge, Lasso, ElasticNet |
| 树模型 | DecisionTree, RandomForest, GradientBoosting, XGBoost |
| 神经网络 | MLP, LSTM, GRU, Transformer |
| 外部模型 | LEAR (epftoolbox) |

### 5. 模型评估

```bash
# 评估所有已训练模型
python main.py evaluate

# 评估指定模型
python main.py evaluate --models Lasso XGBoost
```

### 6. 进行预测

```bash
python main.py predict --model Lasso --date 2025-04-01
```

### 7. 超参数调优

```bash
python main.py tune --model XGBoost --method grid
```

调优方法可选：`grid`（网格搜索）、`random`（随机搜索）、`bayesian`（贝叶斯优化）

## 详细使用指南

### 数据流程

本项目采用分层数据管理策略，避免重复处理：

```
raw/原始数据.csv
    ↓ preprocess
processed/清洗后数据.csv
    ↓ features
features/特征数据.csv
    ↓ train
训练好的模型
```

**关键特点：**
- 预处理和特征工程只需执行一次
- 训练时直接从features加载，速度极快
- 每个阶段数据持久化，便于检查和分析

### 特征说明

系统自动生成以下特征：

1. **时间特征**
   - 月份、星期、是否周末
   - 是否高峰时段（8-22点）
   - 正弦/余弦编码的小时特征

2. **滞后特征**
   - 价格滞后1/2/3/7/14天

3. **滚动统计特征**
   - 7天滚动均值、标准差

4. **目标变量**
   - 24小时的价格序列（Price_H00 ~ Price_H23）

### 模型配置

在 `code/config.py` 中可以修改：

```python
# 数据划分比例
split_config = {
    'test_size_days': 28,      # 测试集28天
    'validation_size_days': 14 # 验证集14天
}

# 特征工程配置
feature_config = {
    'lag_periods': [1, 2, 3, 7],
    'rolling_windows': [7, 14, 30]
}
```

### 评估指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Square Error): 均方根误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差
- **sMAPE** (Symmetric MAPE): 对称平均绝对百分比误差

## 命令行参考

### 主程序命令

```bash
# 数据预处理
python main.py preprocess

# 特征工程
python main.py features

# 训练模型
python main.py train [--models MODEL1 MODEL2 ...]

# 评估模型
python main.py evaluate [--models MODEL1 MODEL2 ...] [--no-viz]

# 预测
python main.py predict --model MODEL [--date YYYY-MM-DD]

# 超参数调优
python main.py tune --model MODEL --method {grid,random,bayesian}

# 列出所有可用模型
python main.py list
```

### 示例工作流

```bash
# 完整流程示例
cd code/

# 1. 准备数据（只需一次）
python main.py preprocess
python main.py features

# 2. 训练并评估线性模型
python main.py train --models Lasso Ridge ElasticNet
python main.py evaluate --models Lasso Ridge ElasticNet

# 3. 训练树模型
python main.py train --models RandomForest XGBoost
python main.py evaluate --models RandomForest XGBoost

# 4. 选择最佳模型进行预测
python main.py predict --model XGBoost --date 2025-04-01
```

## 常见问题

### Q: 训练时提示特征数据不存在？
A: 请先运行 `python main.py preprocess` 和 `python main.py features`

### Q: 如何添加新的特征？
A: 修改 `code/feature_engineering.py` 中的 `create_all_features()` 方法，然后重新运行 `python main.py features`

### Q: 模型训练出现 ConvergenceWarning？
A: 这是正常现象，表示模型在默认迭代次数内未完全收敛。已通过特征标准化优化，通常不影响最终效果。

### Q: 如何调整训练/验证/测试集比例？
A: 修改 `code/config.py` 中的 `split_config` 配置

### Q: 支持自定义模型吗？
A: 支持。继承 `models/base_model.py` 中的 `BaseModel` 类，实现 `fit()` 和 `predict()` 方法即可。

## 项目文档

- [电价预测实验方案](电价预测实验方案.md) - 详细的实验设计方案
- [EPF工具箱使用文档](EPF工具箱使用文档.md) - LEAR模型使用说明
- [电价预测数据分析报告](电价预测数据分析报告.md) - 数据分析报告

## 开发团队

本项目用于湖北省电力现货市场日前电价预测研究。

## 许可证

MIT License
