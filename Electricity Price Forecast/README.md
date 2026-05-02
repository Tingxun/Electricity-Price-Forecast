# 湖北省日前电价预测系统

基于 Direct 多步预测策略的日前电价预测项目。当前框架已删除旧的 global / 多输出策略，只保留“每个预测小时一个独立模型”的主实验流程。

## 项目结构

```text
Electricity Price Forecast/
├── code/
│   ├── main.py                    # Direct 工作流统一入口
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py              # 路径与默认训练配置
│   ├── data_split.py              # 按自然月划分训练/测试
│   ├── feature_selector.py        # 内置特征组与模型特征选择
│   ├── feature_engineering_direct.py
│   ├── model_factory.py           # 轻量模型注册/创建
│   ├── strategy_registry.py       # 预测策略注册与未来策略预留
│   ├── train_direct.py            # 每小时独立训练和调参
│   ├── evaluate_direct.py         # Direct 模型评估
│   ├── backtest_direct.py         # 月份滚动回测
│   ├── predict_direct.py          # 24 小时预测
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/direct/           # H00-H23 独立特征文件
├── saved_models/direct/           # H00-H23 独立模型
├── results/
│   ├── logs/
│   ├── predictions/
│   └── figures/
└── Exploratory Analysis/          # 探索性分析 notebook
```

## 策略说明

Direct 策略将 24 小时价格预测拆成 24 个独立监督学习任务：

```text
H00: f_00(X_00) -> Price_H00
H01: f_01(X_01) -> Price_H01
...
H23: f_23(X_23) -> Price_H23
```

这样每个小时可以单独选择特征、单独搜索超参数、单独保存模型和评估误差。旧的 global 多输出策略已删除，避免保留一个“接口上像 MIMO、内部仍近似 Direct”的冗余流程。

当前代码只有 `direct` 策略可运行，但 CLI 和框架已预留策略层：

```text
direct      已实现：24 个独立单输出模型
recursive  预留：一步模型迭代预测，后续步长使用前序预测值
mimo        预留：真正联合输出 24 小时曲线的多输出模型
```

`code/strategy_registry.py` 负责登记策略状态。未来新增策略时，推荐新增独立文件，例如 `feature_engineering_recursive.py`、`train_recursive.py`、`evaluate_recursive.py`，而不是把不同策略混进 Direct 脚本。

## 依赖

```bash
pip install pandas numpy scikit-learn joblib
pip install lightgbm xgboost
```

`lightgbm` 和 `xgboost` 只在训练对应模型时需要。线性基线 `ridge` / `lasso` 只依赖 scikit-learn。

## 快速开始

在项目根目录或 `code/` 目录运行均可。

### 1. 生成 Direct 特征

```bash
python code/main.py features --strategy direct
```

输入文件：

```text
data/processed/processed_data.csv
```

输出文件：

```text
data/features/direct/features_H00.csv
...
data/features/direct/features_H23.csv
```

### 2. 训练模型

```bash
# 训练 LightGBM，每小时随机搜索 20 组参数
python code/main.py train --strategy direct --model lightgbm --n-iter 20

# 只训练指定小时
python code/main.py train --strategy direct --model lightgbm --hours 0 8 12 18 --n-iter 20

# 使用默认参数，不做随机搜索
python code/main.py train --strategy direct --model lightgbm --n-iter 0

# 指定一个或多个自然月作为测试集；不指定时默认使用最后一个可用月份
python code/main.py train --strategy direct --model lightgbm --test-months 2025-03 --n-iter 0
python code/main.py train --strategy direct --model lightgbm --test-months 2025-02 2025-03 --n-iter 0
```

支持的模型：

```text
lightgbm
lightgbm_smape_probe
xgboost
random_forest
ridge
lasso
```

`lightgbm_smape_probe` 是本轮 sMAPE 探针实验沉淀的 LightGBM 参数组：每个小时可以使用不同 objective / quantile alpha。它用于复现实验结果，不会覆盖标准 `lightgbm`。

模型保存到：

```text
saved_models/direct/{model}/model_H00.pkl
saved_models/direct/{model}/metadata_H00.json
...
```

### 3. 评估模型

```bash
python code/main.py evaluate --strategy direct --model lightgbm
python code/main.py evaluate --strategy direct --model lightgbm --test-months 2025-03
python code/main.py evaluate --strategy direct --model lightgbm --test-months 2025-02 2025-03
```

评估报告输出到：

```text
results/logs/direct/{model}/evaluation_report.csv
results/predictions/direct/{model}/test_predictions.csv
```

### 4. 预测

```bash
python code/main.py predict --strategy direct --model lightgbm --date 2025-03-26
```

如果指定日期在已生成特征中不存在，预测脚本会使用最新一行特征，并在日志中提示。

### 5. 一键流程

```bash
python code/main.py run-all --strategy direct --model lightgbm --n-iter 20
```

### 6. 月份滚动回测

滚动回测采用扩展窗口：测试某个月时，只使用该月之前的全部月份训练。

```bash
python code/main.py backtest --strategy direct --model lightgbm --n-iter 0 --min-train-months 3
python code/main.py backtest --strategy direct --model lightgbm_smape_probe --n-iter 0 --min-train-months 3
```

输出文件：

```text
results/logs/direct/{model}/monthly_backtest.csv
results/logs/direct/{model}/monthly_backtest_summary.csv
results/logs/direct/{model}/monthly_backtest_overall.json
```

## 特征控制

所有模型输入特征由 `code/feature_selector.py` 中的 `FEATURE_CONFIG` 控制。配置支持精确列名和正则模式，例如：

```python
"direct_market_window": {
    "patterns": [
        r"^(当前|滞后1h|未来1h)_市场_",
        r"^市场变化_",
        r"^市场日形态_",
    ],
}
```

给某个模型换特征时，修改 `FEATURE_CONFIG["model_features"]` 下对应模型的 `feature_groups`、`include_patterns` 或 `exclude_patterns`。

当前特征工程不使用日前价格；市场输入来自预处理后的可用市场边界字段，并额外构造净负荷、新能源、市场化缺口、供需覆盖率和相邻小时爬坡等特征。

## 推荐实验顺序

```bash
python code/main.py features --strategy direct
python code/main.py train --strategy direct --model ridge --n-iter 0
python code/main.py evaluate --strategy direct --model ridge
python code/main.py train --strategy direct --model lightgbm --n-iter 20
python code/main.py evaluate --strategy direct --model lightgbm
```

先用 `ridge` 快速验证数据和流程，再用 `lightgbm` 做主实验，会比较省时间。

复现实验探针结果：

```bash
python code/main.py train --strategy direct --model lightgbm_smape_probe --n-iter 0
python code/main.py evaluate --strategy direct --model lightgbm_smape_probe
```
