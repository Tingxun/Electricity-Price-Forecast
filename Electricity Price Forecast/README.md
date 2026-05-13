# Electricity Price Forecast

基于多源异构数据的日前电价预测项目。当前主流程支持 Direct 24 小时独立建模和 MIMO 多输出建模，统一入口已迁移到 `EPF` 包。

## 项目结构

```text
.
├─ src/EPF/                 # 可导入的 Python 包
│  ├─ cli.py                # 统一命令行入口
│  ├─ config/               # 路径与实验配置
│  ├─ feature_engineering/  # 特征工程与特征选择代码
│  ├─ models/               # 模型工厂与模型定义
│  ├─ strategies/           # Direct/MIMO 训练、评估、预测、回测
│  ├─ reports/              # 模型对比与报告生成
│  └─ utils/                # 指标、切分、评估、模型存储等工具
├─ tests/                   # 自动化测试
├─ notebooks/               # EDA 和数据预处理 Notebook
├─ docs/                    # 项目报告与技术文档
├─ data/                    # 数据目录；data/features 是生成后的特征数据产物
├─ results/                 # 本地运行结果，默认不入库
└─ saved_models/            # 本地模型产物，默认不入库
```

## 安装

建议在项目根目录安装为可编辑包：

```bash
pip install -e .
```

安装后可以直接使用命令：

```bash
epf list
```

如果不安装，也可以临时指定源码路径：

```powershell
$env:PYTHONPATH="src"; python -m EPF.cli list
```

## 基本流程

生成特征：

```bash
epf features --strategy direct
```

固定参数训练：

```bash
epf train --strategy direct --model lightgbm_auto --test-months 2025-03 --fixed-params
```

自动结构选择与调参：

```bash
epf train --strategy direct --model lightgbm_auto --test-months 2025-03 --n-iter 20 --cv-folds 3
```

只训练部分小时：

```bash
epf train --strategy direct --model lightgbm_auto --test-months 2025-03 --hours 0 1 2
```

评估指定测试月模型：

```bash
epf evaluate --strategy direct --model lightgbm_auto --test-months 2025-03
```

预测指定日期：

```bash
epf predict --strategy direct --model lightgbm_auto --date 2025-04-01 --test-months 2025-03
```

完整执行特征、训练、评估：

```bash
epf run-all --strategy direct --model lightgbm_auto --test-months 2025-03 --n-iter 20
```

## 模型保存

训练产物按策略、模型和测试期版本化保存：

```text
saved_models/direct/<model_type>/<test_period>/
saved_models/mimo/<model_type>/<test_period>/
```

Direct 每个小时通常包含：

- `model_Hxx.pkl`
- `metadata_Hxx.json`
- `feature_importance/feature_importance_Hxx.csv`

运行级别通常包含：

- `manifest.json`
- `best_params_by_hour.json`
- `training_report.csv`
- `structure_search_results.csv`
- `hyperparameter_search_results.csv`
