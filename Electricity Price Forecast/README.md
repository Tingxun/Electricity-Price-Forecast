# Electricity Price Forecast

本项目采用 Direct 24 小时独立建模框架预测日前电价：每个目标小时训练一个独立模型。

当前推荐正式训练入口是 `lightgbm_auto`。训练时会把目标测试月之前的全部数据作为训练窗口，在训练窗口内用按月滚动的时间序列交叉验证选择结构/特征组，再对胜出结构做超参数搜索，最后用完整训练窗口重训并保存模型。测试月只用于最终评估，不参与结构选择或参数搜索。

## 可用模型

```bash
python code/main.py list
```

主要模型：

- `lightgbm_auto`：自动结构/特征组选择 + 可选超参数搜索的推荐模型。
- `lightgbm`：LightGBM 固定结构基线。
- `xgboost`、`random_forest`、`ridge`、`lasso`：对照基线。

## 基本流程

生成特征：

```bash
python code/main.py features --strategy direct
```

固定参数训练：

```bash
python code/main.py train --strategy direct --model lightgbm_auto --test-months 2025-03 --fixed-params
```

自动结构选择并调参：

```bash
python code/main.py train --strategy direct --model lightgbm_auto --test-months 2025-03 --n-iter 20 --cv-folds 3
```

只训练部分小时：

```bash
python code/main.py train --strategy direct --model lightgbm_auto --test-months 2025-03 --hours 0 1 2
```

评估指定测试月模型：

```bash
python code/main.py evaluate --strategy direct --model lightgbm_auto --test-months 2025-03
```

预测时可指定使用哪个测试月训练出的模型；不指定时默认读取最新保存版本：

```bash
python code/main.py predict --strategy direct --model lightgbm_auto --date 2025-04-01 --test-months 2025-03
```

完整执行特征、训练、评估：

```bash
python code/main.py run-all --strategy direct --model lightgbm_auto --test-months 2025-03 --n-iter 20
```

## 模型保存

训练产物按模型和测试期版本化保存：

```text
saved_models/direct/<model_type>/<test_period>/
```

每个小时保存：

- `model_Hxx.pkl`
- `metadata_Hxx.json`
- `feature_importance/feature_importance_Hxx.csv`

运行级别保存：

- `manifest.json`
- `best_params_by_hour.json`
- `training_report.csv`
- `structure_search_results.csv`
- `hyperparameter_search_results.csv`

`metadata_Hxx.json` 会记录测试期、训练窗口、训练模式、入选结构、入选特征组、候选结构排名、CV 指标、最佳参数和最终测试月指标。
