rrA# EPF工具箱使用文档

## 概述

EPF工具箱（epftoolbox）是一个开源的电力价格预测研究工具库，旨在为电力价格预测研究提供可复现的工具和基准。该库基于scikit-learn、tensorflow、keras、hyperopt、statsmodels、numpy和pandas构建。

### 主要特性

- **预测模型**: 包含两种最先进的预测模型（深度神经网络DNN和LASSO自回归模型LEAR）
- **评估指标**: 提供多种标准评估指标和统计检验方法
- **数据预处理**: 内置数据缩放和预处理功能
- **超参数优化**: 支持自动超参数优化
- **基准测试**: 提供标准化的基准测试框架

## 安装指南

### 系统要求

- Python 3.9-3.13（64位版本）
- 支持的操作系统：Windows、macOS、Linux

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/jeslago/epftoolbox.git
cd epftoolbox
```

2. **创建虚拟环境（推荐）**
```bash
# 使用conda
conda create --name epftoolbox python=3.10
conda activate epftoolbox

# 或使用venv
python -m venv epftoolbox-env
source epftoolbox-env/bin/activate  # Linux/macOS
epftoolbox-env\Scripts\activate    # Windows
```

3. **安装依赖**
```bash
pip install .
```

### 验证安装

```python
import epftoolbox
print("EPF工具箱安装成功！")
```

## 核心模块介绍

### 1. 数据模块 (epftoolbox.data)

#### 数据读取
```python
from epftoolbox.data import read_data

# 读取标准数据集
data = read_data(dataset='PJM', path='./datasets')
```

#### 数据预处理
```python
from epftoolbox.data import scaling, DataScaler

# 数据缩放
scaler = DataScaler()
scaled_data = scaler.fit_transform(data)
```

### 2. 预测模型模块 (epftoolbox.models)

#### LEAR模型
```python
from epftoolbox.models import LEAR, evaluate_lear_in_test_dataset

# 创建LEAR模型
lear_model = LEAR()

# 在测试数据集上评估
results = evaluate_lear_in_test_dataset(
    dataset='PJM',
    years_test=2,
    calibration_window=364*4
)
```

#### DNN模型
```python
from epftoolbox.models import DNN, DNNModel, evaluate_dnn_in_test_dataset

# 创建DNN模型
dnn_model = DNN(nlayers=2)

# 在测试数据集上评估
results = evaluate_dnn_in_test_dataset(
    dataset='PJM',
    years_test=2,
    nlayers=2,
    calibration_window=4
)
```

### 3. 评估模块 (epftoolbox.evaluation)

#### 评估指标
```python
from epftoolbox.evaluation import MAE, RMSE, MAPE, sMAPE, MASE, rMAE

# 计算各种评估指标
mae = MAE(p_real, p_pred)
rmse = RMSE(p_real, p_pred)
mape = MAPE(p_real, p_pred)
smape = sMAPE(p_real, p_pred)
mase = MASE(p_real, p_pred, p_naive)
rmae = rMAE(p_real, p_pred, p_naive)
```

#### 统计检验
```python
from epftoolbox.evaluation import DM, GW

# Diebold-Mariano检验
dm_test = DM(p_real, p_pred1, p_pred2)

# Giacomini-White检验
gw_test = GW(p_real, p_pred1, p_pred2)
```

## 使用教程

### 基础使用示例

#### 示例1：使用LEAR模型进行预测

```python
import os
from epftoolbox.models import evaluate_lear_in_test_dataset

# 配置参数
dataset = 'PJM'
years_test = 2
calibration_window = 364 * 4

# 设置路径
path_datasets_folder = os.path.join('.', 'datasets')
path_recalibration_folder = os.path.join('.', 'experimental_files')

# 执行评估
results = evaluate_lear_in_test_dataset(
    path_recalibration_folder=path_recalibration_folder,
    path_datasets_folder=path_datasets_folder,
    dataset=dataset,
    years_test=years_test,
    calibration_window=calibration_window
)

print("LEAR模型评估完成！")
```

#### 示例2：使用DNN模型进行预测

```python
import os
from epftoolbox.models import evaluate_dnn_in_test_dataset

# 配置参数
dataset = 'PJM'
years_test = 2
nlayers = 2
calibration_window = 4

# 设置路径
path_datasets_folder = os.path.join('.', 'datasets')
path_recalibration_folder = os.path.join('.', 'experimental_files')
path_hyperparameter_folder = os.path.join('.', 'experimental_files')

# 执行评估
results = evaluate_dnn_in_test_dataset(
    path_recalibration_folder=path_recalibration_folder,
    path_datasets_folder=path_datasets_folder,
    path_hyperparameter_folder=path_hyperparameter_folder,
    dataset=dataset,
    years_test=years_test,
    nlayers=nlayers,
    calibration_window=calibration_window,
    shuffle_train=1,
    data_augmentation=0,
    new_recalibration=1,
    experiment_id=1
)

print("DNN模型评估完成！")
```

### 高级功能

#### 超参数优化

```python
from epftoolbox.models import hyperparameter_optimizer

# 执行超参数优化
best_params = hyperparameter_optimizer(
    dataset='PJM',
    nlayers=2,
    calibration_window=4,
    experiment_id=1
)

print("最优超参数：", best_params)
```

#### 模型比较

```python
from epftoolbox.evaluation import DM, plot_multivariate_DM_test

# 比较两个模型的预测性能
dm_results = DM(p_real, p_pred_lear, p_pred_dnn)

# 绘制多变量DM检验结果
plot_multivariate_DM_test(dm_results)
```

## 数据格式要求

### 输入数据格式

EPF工具箱期望的数据格式为CSV文件，包含以下列：

- `Date`: 日期时间戳（格式：YYYY-MM-DD HH:MM:SS）
- `Price`: 电力价格数据
- 可选的其他特征列（如负荷、温度等）

### 标准数据集

工具箱预定义了多个标准数据集：
- `PJM`: 美国PJM市场数据
- `NP`: 北欧电力市场数据
- `BE`: 比利时电力市场数据
- `FR`: 法国电力市场数据
- `DE`: 德国电力市场数据

## 最佳实践

### 1. 数据预处理

- 确保数据没有缺失值
- 对异常值进行适当处理
- 使用内置的缩放功能进行数据标准化

### 2. 模型选择

- **LEAR模型**: 适合线性关系较强的数据，计算效率高
- **DNN模型**: 适合复杂非线性关系，需要更多计算资源

### 3. 参数调优

- 使用`calibration_window`参数控制训练窗口大小
- 对于DNN模型，合理设置`nlayers`和超参数
- 使用内置的超参数优化功能

### 4. 评估策略

- 使用多种评估指标进行综合评估
- 进行统计检验确保结果显著性
- 考虑不同时间尺度的预测性能

## 故障排除

### 常见问题

#### 1. 安装问题

**问题**: TensorFlow安装失败
**解决方案**: 确保使用支持的Python版本（3.9-3.13），并安装64位Python

**问题**: 依赖冲突
**解决方案**: 使用虚拟环境隔离依赖

#### 2. 运行问题

**问题**: 内存不足
**解决方案**: 减小`calibration_window`参数或使用数据子集

**问题**: 数据集找不到
**解决方案**: 确保数据集文件位于正确路径，或提供完整文件路径

#### 3. 性能问题

**问题**: 训练速度慢
**解决方案**: 使用GPU加速（如果可用），或减小模型复杂度

### 调试技巧

1. **启用详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查数据完整性**
```python
print("数据形状：", data.shape)
print("缺失值数量：", data.isnull().sum())
```

3. **验证模型配置**
```python
print("模型参数：", model.get_config())
```

## 扩展开发

### 添加新模型

要添加新的预测模型，需要：

1. 在`epftoolbox/models`目录下创建新模型文件
2. 实现标准的模型接口
3. 在`__init__.py`中导出模型
4. 提供相应的评估函数

### 添加新评估指标

要添加新的评估指标，需要：

1. 在`epftoolbox/evaluation`目录下创建新指标文件
2. 实现指标计算函数
3. 在`__init__.py`中导出指标

## 参考资料

- [官方文档](https://epftoolbox.readthedocs.io/en/latest/)
- [GitHub仓库](https://github.com/jeslago/epftoolbox)
- [相关论文](https://doi.org/10.1016/j.apenergy.2021.116983)

## 许可证

EPF工具箱基于AGPL-3.0许可证发布。使用本工具时请遵守相关许可证条款。

---

*本文档基于EPF工具箱v1.0版本编写，最后更新日期：2026年1月26日*