# 湖北省日前电价预测项目

本项目采用 Direct 24 小时独立建模框架预测日前电价：每个目标小时训练一个独立模型。当前正式模型是 `lightgbm_smape_probe`，实验复现别名是 `lightgbm_smape_probe_midday_v3`。

详细优化过程、参数调优、特征工程和模型集成说明见 [lightgbm_smape_probe优化技术文档.md](./lightgbm_smape_probe优化技术文档.md)。

## 项目结构

```text
code/main.py                         统一 CLI 入口
code/feature_engineering_direct.py   Direct 特征工程
code/feature_selector.py             特征组与模型特征路由
code/model_factory.py                模型注册、参数与集成封装
code/probe_optimizer.py              LightGBM 探针优化
code/train_direct.py                 训练入口
code/evaluate_direct.py              评估入口
code/backtest_direct.py              月份滚动回测
data/features/direct/                每小时特征文件 features_Hxx.csv
saved_models/direct/                 训练后的每小时模型
results/logs/direct/                 评估、回测和探针日志
results/predictions/direct/          测试集预测结果
```

## 可用模型

```text
lightgbm
lightgbm_smape_probe
lightgbm_smape_probe_midday_v3
xgboost
random_forest
ridge
lasso
```

说明：

- `lightgbm_smape_probe`：当前正式模型，已集成午间 v3 优化。
- `lightgbm_smape_probe_midday_v3`：与正式模型同配置，用于复现实验命名。
- `lightgbm`：普通 LightGBM 基线，不包含 sMAPE 探针参数和午间集成。

## 快速运行

生成 Direct 特征：

```bash
python code/main.py features --strategy direct
```

训练正式模型：

```bash
python code/main.py train --strategy direct --model lightgbm_smape_probe --test-months 2025-03 --n-iter 0
```

评估正式模型：

```bash
python code/main.py evaluate --strategy direct --model lightgbm_smape_probe --test-months 2025-03
```

滚动回测：

```bash
python code/main.py backtest --strategy direct --model lightgbm_smape_probe --n-iter 0 --min-train-months 3
```

运行探针优化：

```bash
python code/main.py optimize-probe --model lightgbm_smape_probe_midday_v3 --hours 8 9 10 11 12 13 14 15 --test-months 2025-03 --max-candidates 180
```

## 当前结果

```text
2025-03 正式 lightgbm_smape_probe:
平均 MAE   = 50.8029
平均 RMSE  = 76.1840
平均 sMAPE = 24.58%
under20    = 13/24
H08-H15 平均 sMAPE = 36.10%
```

滚动回测：

```text
2024-09 至 2025-03 overall sMAPE = 38.69%
```

## 文档索引

- [电价预测实验状态.md](./电价预测实验状态.md)：当前实验状态和复现命令。
- [lightgbm_smape_probe优化技术文档.md](./lightgbm_smape_probe优化技术文档.md)：完整优化过程和技术说明。
- [电价预测数据分析报告.md](./电价预测数据分析报告.md)：数据探索分析报告。
