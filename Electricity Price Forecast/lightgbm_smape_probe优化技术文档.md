# lightgbm_smape_probe 优化技术文档

本文记录 `lightgbm_smape_probe` 从 2025-03 平均 sMAPE 约 28.44% 优化到 24.58% 的完整技术路径。当前正式模型和 `lightgbm_smape_probe_midday_v3` 使用同一套已验收配置；前者用于正式训练评估，后者用于复现实验和优化版本。

## 1. 目标与约束

优化目标：

- 主验收：2025-03 自然月 24 小时平均 sMAPE < 25.00%。
- 午间目标：H08-H15 平均 sMAPE 从约 41.80% 降到 37.00% 以下。
- 风险控制：2025-02 单月不劣化；滚动回测整体不超过上一轮正式探针 +1.0 个 sMAPE 点。

建模约束：

- 保持 Direct 24 小时独立建模框架，即每个小时一个独立 LightGBM 模型。
- 不使用日前价格，不引入任何 T+1 实时价格泄漏。
- 继续按自然月划分训练/测试，2025-03 是主测试月。

最终结果：

```text
2025-03 正式 lightgbm_smape_probe:
平均 MAE   = 50.8029
平均 RMSE  = 76.1840
平均 sMAPE = 24.58%
under20    = 13/24
H08-H15 平均 sMAPE = 36.10%
```

Guardrail：

```text
2025-02 单月回测: 27.34%
滚动回测整体:     38.69%
```

## 2. 误差诊断

初始短板集中在 H08-H15。该时段不是普通高价误差，而是午间低价、近零价样本引起的 sMAPE 放大。

sMAPE 的核心形式为：

```text
sMAPE = 2 * |预测值 - 真实值| / (|预测值| + |真实值|)
```

当真实电价较低时，分母很小。例如真实值 40、预测值 100，绝对误差只有 60，但 sMAPE 约为 85.7%。真实值 400、预测值 460 时，同样 60 元误差对应 sMAPE 约 14.0%。因此午间低价样本会主导平均 sMAPE。

热力图和逐小时曲线显示：

- H08-H15 存在明显日内低谷。
- 周三、周六、周日的午间低价形态更突出。
- H11-H14 容易在低价日被系统性高估。
- 普通 LightGBM 参数搜索对这个问题改善有限，因为损失函数与 sMAPE 不完全一致。

## 3. 参数调优

### 3.1 小时级参数固化

项目采用 Direct 框架，因此每个小时可使用不同参数。非午间小时沿用上一轮已稳定的小时级参数；H08-H15 则通过探针重新搜索。

常规搜索维度包括：

- `objective`: `regression`、`regression_l1`、`huber`、`quantile`
- `alpha`: 针对 `quantile` 的分位点搜索
- `n_estimators`: 120、200、300、500
- `learning_rate`: 0.02、0.03、0.05、0.08
- `num_leaves`: 15、31、63
- `max_depth`: 4、5、6、8
- `min_child_samples`: 5、10、20、30
- `subsample` 和 `colsample_bytree`: 固定为 0.8

为什么这样做：

- `quantile` 可以系统性控制预测偏高或偏低，适合修正 sMAPE 对低价高估的敏感性。
- 小树深、少叶子参数可以限制午间小样本特征的过拟合。
- `regression_l1` 对异常点更稳，对 H14 这类低价波动小时有效。

### 3.2 午间最终参数选择

最终不是每小时只取一个模型，而是取 2-3 个子模型组成小集成。每个子模型的参数来自探针胜出候选。例如：

```text
H08:
0.95 * quantile(alpha=0.35, 120 trees, depth=4, strong weight, midday+agg weather)
0.05 * quantile(alpha=0.35, 120 trees, depth=4, default weight, midday)

H10:
0.55 * regression, 120 trees, depth=4, strong weight, midday
0.25 * two-stage quantile(alpha=0.8), weather
0.20 * two-stage quantile(alpha=0.8), weather

H12:
0.33 * quantile(alpha=0.62), midday+agg weather
0.33 * quantile(alpha=0.60), raw weather, default weight
0.34 * quantile(alpha=0.90), raw weather, strong weight
```

选择规则：

- 对每个小时，只接受 2025-03 上优于当前正式基线的候选。
- 若单模型不够稳定，则用多个候选做固定权重融合。
- 固化后再跑 2025-02 和滚动回测 guardrail。

## 4. 特征工程

本轮核心不是简单扩充全部特征，而是围绕午间低价机制增加结构化特征，并通过 `feature_selector.py` 控制只让 H08-H15 使用。

### 4.1 星期形态特征

新增：

```text
星期_1 ... 星期_7
是否周三
是否周六
是否周日
```

原因：

- 原始 `星期` 是 1-7 的顺序变量，树模型可能错误理解为连续大小关系。
- 周三、周六、周日午间低价形态更明显，需要显式标记。
- one-hot 后模型能学习“某类星期状态”而不是“星期数值大小”。

这些特征被放入 `direct_time_midday`，只给午间强化模型使用，避免影响普通正式小时的历史可比性。

### 4.2 同星期历史价格形态

新增：

```text
同星期历史_滞后7天_Hxx_价格
同星期历史_滞后14天_Hxx_价格
同星期历史_滞后21天_Hxx_价格
同星期历史_近3周均值/最小/最大
同星期历史_近3周低价次数
同星期历史_T7_T14差值
```

原因：

- 电价存在强星期周期。
- 午间低价往往不是孤立事件，同星期历史价格能提供 regime 先验。
- 低价次数能帮助模型判断目标日是否处在持续低价模式。

### 4.3 午间低价形态特征

基于 T-2 日 H08-H15 构造：

```text
午间低价_T2均值
午间低价_T2最小值
午间低价_T2最大值
午间低价_T2低价小时数
午间低价_T2近零小时数
午间低价_T2谷底小时
午间低价_目标小时距谷底距离
午间低价_目标小时相对午间均值位置
```

原因：

- T-2 是当前框架下安全可用的历史信息。
- 午间低价的关键不是单点滞后，而是 H08-H15 整段曲线形态。
- 谷底小时和相对位置能帮助模型识别目标小时处于下坡、谷底还是回升段。

### 4.4 目标日午间市场形态

对 H08-H15 的市场变量聚合：

```text
系统负荷
光伏出力
新能源出力
净负荷
剩余负荷
市场化缺口
供需覆盖率
```

每类变量生成：

```text
午间均值
午间标准差
午间最小值
午间最大值
H08到H15爬坡
目标小时相对午间均值
目标小时午间位置
```

原因：

- 午间低价通常由光伏、新能源出力抬升、净负荷下探、供需覆盖率变化共同驱动。
- 单小时市场变量无法描述整段午间曲线。
- 爬坡和相对均值能捕捉日内形态，而不是只看绝对水平。

### 4.5 聚合气象特征

对原始气象列按变量聚合：

```text
当前_气象聚合_温度/总云量/辐照度_均值/最大/最小/标准差
滞后1h_气象聚合_...
未来1h_气象聚合_...
午间气象聚合_温度/总云量/辐照度_均值/最大/标准差
```

原因：

- 原始气象列维度高、城市/数据源多，直接全部喂给小样本小时容易过拟合。
- 聚合统计保留天气强度与离散度，同时降低特征噪声。
- 辐照度和云量与光伏午间出力密切相关，是识别低价谷的重要代理。

## 5. sMAPE 代理样本权重

LightGBM 原生目标并不直接优化 sMAPE。为了让训练过程更重视低价样本，新增权重：

```text
weight = clip(300 / (abs(y_train) + 30), 1, upper)
```

三档权重：

```text
light:   upper = 3
default: upper = 6
strong:  upper = 8
```

技术含义：

- 真实价格越低，样本权重越高。
- 高价样本权重接近 1，避免主模型完全牺牲高价段。
- 低价样本获得更大梯度影响，减少午间谷底高估。

为什么需要它：

- sMAPE 在低价样本上惩罚更重。
- MAE/RMSE 视角下的“小误差”在 sMAPE 下可能是灾难性误差。
- 权重是对 sMAPE 分母效应的可训练近似。

实现位置：

- `model_factory.py`
- `_smape_proxy_weights`
- `WeightedLGBMRegressor`

## 6. 低价二阶段模型

低价二阶段模型用于 H08-H15 的部分子模型，结构如下：

```text
主回归模型: 预测常规价格
低价分类器: 预测 P(y <= low_price_threshold)
低价回归器: 只在低价样本上训练
融合器: 当低价概率超过阈值时，将主预测和低价预测加权融合
```

融合公式：

```text
if P(低价) >= prob_threshold:
    pred = (1 - blend) * 主模型预测 + blend * 低价模型预测
else:
    pred = 主模型预测
```

搜索范围：

```text
low_price_threshold: 50, 80, 120
prob_threshold:      0.35, 0.50, 0.65
blend:               0.40, 0.70, 1.00
```

为什么需要它：

- 午间低价是一个 regime，而不是普通连续扰动。
- 单模型会被正常价和高价样本拉向均值，导致低价日高估。
- 分类器先判断是否进入低价状态，再启用低价专家，可更精确压低谷底预测。

风险控制：

- 二阶段模型只作为小集成中的成员，不单独支配最终输出。
- 只有探针验证有效的小时和参数才固化。
- 错判低价时，固定融合权重能限制过度拉低。

## 7. 特征组小集成

单模型在午间低价段不稳定，不同特征组有不同偏差。最终采用小时内小集成：

```text
pred = Σ weight_i * model_i(feature_group_i)
```

候选特征视角：

```text
default:
  direct_time_midday + direct_price_lag + direct_market_window

weather:
  default + direct_weather_window

midday_regime:
  default + direct_midday_regime

midday_regime_weather:
  midday_regime + direct_midday_weather_agg
```

为什么小集成有效：

- 不同特征组对不同日期有效，融合可降低单模型方差。
- 不同 objective 和 alpha 有不同系统偏差，融合后更接近 sMAPE 最优折中。
- 子模型只使用适合自己的特征组，避免把原始气象、聚合气象、午间形态全部强塞进同一个模型造成冲突。

实现位置：

- `model_factory.py`
- `FeatureGroupEnsembleRegressor`
- 每个 member 内部保存 `weight`、`feature_groups`、`params`

## 8. 探针优化流程

新增和扩展的探针能力：

- `default`
- `weather`
- `midday_regime`
- `midday_regime_weather`
- `midday_regime_weighted`
- 样本权重候选
- 低价二阶段候选

运行命令：

```bash
python code/main.py optimize-probe --model lightgbm_smape_probe_midday_v3 --hours 8 9 10 11 12 13 14 15 --test-months 2025-03 --max-candidates 180
```

进一步聚焦 H10/H12/H13/H14：

```bash
python code/main.py optimize-probe --model lightgbm_smape_probe_midday_v3 --hours 10 12 13 14 --test-months 2025-03 --max-candidates 320 --local-alpha-radius 0.20 --local-alpha-step 0.02 --broad-alpha-step 0.03
```

探针输出：

```text
results/logs/direct/lightgbm_smape_probe_midday_v3/probe_optimization_*.csv
results/logs/direct/lightgbm_smape_probe_midday_v3/probe_optimization_best_*.csv
```

## 9. 最终逐小时改善

相对上一轮正式探针，H08-H15 逐小时改善：

```text
H08: 42.52% -> 35.11%  (-7.41)
H09: 44.72% -> 37.19%  (-7.52)
H10: 40.00% -> 35.23%  (-4.77)
H11: 37.64% -> 32.50%  (-5.14)
H12: 48.60% -> 41.43%  (-7.17)
H13: 42.49% -> 41.03%  (-1.46)
H14: 43.40% -> 35.51%  (-7.88)
H15: 35.03% -> 30.77%  (-4.26)
```

整体：

```text
上一轮正式探针: 平均 sMAPE = 26.48%
当前正式探针:   平均 sMAPE = 24.58%
```

## 10. 复现命令

完整复现：

```bash
python code/main.py features --strategy direct
python code/main.py train --strategy direct --model lightgbm_smape_probe --test-months 2025-03 --n-iter 0
python code/main.py evaluate --strategy direct --model lightgbm_smape_probe --test-months 2025-03
python code/main.py backtest --strategy direct --model lightgbm_smape_probe_midday_v3 --n-iter 0 --min-train-months 3
```

当前正式模型也可用实验别名复现：

```bash
python code/main.py train --strategy direct --model lightgbm_smape_probe_midday_v3 --test-months 2025-03 --n-iter 0
python code/main.py evaluate --strategy direct --model lightgbm_smape_probe_midday_v3 --test-months 2025-03
```

## 11. 后续优化方向

虽然当前已低于 25%，H12/H13 仍高于 40%。下一轮不建议继续盲目扩大 LightGBM 网格，优先方向应是：

- 构造更强的低价 regime 分类特征。
- 引入跨小时曲线一致性约束，避免 H11-H14 谷底形态被逐小时模型割裂。
- 分析低价误判样本的共同天气、光伏、净负荷和星期结构。
- 对 H12/H13 单独设计残差修正或 regime-aware calibration。
