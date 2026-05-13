# Electricity Price Forecast

基于多源异构数据的日前电价预测项目。当前框架以 `src/EPF` 包为统一入口，保留 `direct` 单小时建模基线，同时新增 `mimo` 多输出建模主线，用于对比月度重训、周度重训和更鲁棒的跨月份泛化表现。

## 项目结构

```text
.
|-- src/EPF/
|   |-- cli.py                  # 统一命令行入口
|   |-- schema.py               # 共享列名、时间段和滚动模式常量
|   |-- config/                 # 路径与项目配置
|   |-- feature_engineering/    # Direct 与 MIMO 特征工程
|   |-- models/                 # 模型工厂与神经网络模型
|   |-- strategies/             # 各策略的 train/evaluate/predict/backtest
|   |-- reports/                # 模型对比报告
|   `-- utils/                  # 指标、切分、评估、模型存储、策略注册
|-- tests/                      # 自动化测试
|-- data/                       # 原始、处理后和特征数据
|-- results/                    # 本地运行结果
`-- saved_models/               # 本地模型产物
```

## 安装

建议在项目根目录安装为可编辑包：

```bash
pip install -e .
```

也可以临时指定源码路径运行：

```powershell
$env:PYTHONPATH="src"; python -m EPF.cli list
```

## 常用命令

查看当前可运行策略和模型：

```bash
epf list
```

生成特征：

```bash
epf features --strategy direct
epf features --strategy mimo
```

训练与评估 Direct 基线：

```bash
epf train --strategy direct --model lightgbm_auto --test-months 2025-03 --fixed-params
epf evaluate --strategy direct --model lightgbm_auto --test-months 2025-03
```

训练与评估 MIMO 模型：

```bash
epf train --strategy mimo --model tcn_mimo --test-months 2025-03 --epochs 300
epf evaluate --strategy mimo --model tcn_mimo --test-months 2025-03
```

滚动回测：

```bash
epf backtest --strategy direct --model lightgbm_auto --retrain-frequency monthly --n-iter 0 --min-train-months 3
epf backtest --strategy direct --model lightgbm_auto --retrain-frequency weekly --n-iter 0 --min-train-months 3
epf backtest --strategy mimo --model tcn_mimo --retrain-frequency monthly --min-train-months 3
epf backtest --strategy mimo --model tcn_mimo --retrain-frequency weekly --min-train-months 3
```

## 策略分发

`EPF.utils.strategy_registry` 是 CLI 的实际分发依据。新增策略时需要在注册表中声明：

- `feature_engineer`
- `trainer`
- `evaluator`
- `predictor`
- `backtester`
- `model_types`
- `default_model`

CLI 会通过注册表动态加载组件，而不是在命令入口中硬编码模块路径。

## 产物位置

训练产物按策略、模型和测试期版本化保存：

```text
saved_models/direct/<model_type>/<test_period>/
saved_models/mimo/<model_type>/<test_period>/
```

报告和预测结果默认写入：

```text
results/logs/
results/predictions/
```

## 开发检查

```powershell
$env:PYTHONPATH="src"; python -m unittest discover tests
$env:PYTHONPATH="src"; python -m EPF.cli list
```
