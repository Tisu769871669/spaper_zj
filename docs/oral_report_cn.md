# 小论文中文汇报稿

## 1. 课题题目

基于双层对抗强化学习的网络威胁检测方法研究

英文题目：

Bi-level Adversarial Reinforcement Learning for Robust Network Threat Detection

## 2. 研究背景

当前网络入侵检测面临两个核心问题。

第一，传统 IDS 主要依赖静态特征、历史数据和固定分类模型，当攻击策略发生变化时，模型很容易退化。  
第二，很多已有工作虽然在单一 benchmark 上指标很好，但在更现代的数据集、跨数据集测试或者对抗扰动下，性能会明显下降。

因此，这个课题关注的不是单纯提高一个分类指标，而是研究：

- 能不能把入侵检测建模成 attacker-defender 的交互过程
- 能不能用 bi-level 的方式，让 defender 针对更强的 attacker 响应进行训练

## 3. 研究目标

本课题的目标是提出一个 Bi-level Adversarial Reinforcement Learning，也就是 `Bi-ARL` 框架，用于网络威胁检测。

核心思想是：

- defender 作为 leader
- attacker 作为 follower
- 训练时不是简单同时更新两个 agent
- 而是先让 attacker 朝着更强响应逼近，再让 defender 针对该 attacker 更新

这样做的目的，是希望 defender 学到的策略不仅适应普通样本，还能更好适应对抗式扰动。

## 4. 方法简介

方法整体由四部分组成：

1. 网络安全对抗环境
   - 将流量样本作为状态
   - attacker 对攻击流量施加扰动
   - defender 根据扰动后的观测做判定

2. attacker 和 defender 两个策略网络
   - 都基于 PPO 进行策略优化

3. bi-level 训练机制
   - inner loop：训练 attacker
   - outer loop：固定 attacker，训练 defender

4. 评测与对比体系
   - 主指标使用 Recall、Precision、F1、FPR
   - 对比普通 RL、MARL、LSTM 和强表格模型

## 5. 数据集与实验设置

目前已经完成三类数据集实验：

### 5.1 NSL-KDD

这是相对传统、较受控的数据集。  
它的作用主要是验证 `Bi-ARL` 这条 RL 方法线是否成立。

### 5.2 UNSW-NB15

这是更现代的网络入侵检测数据集。  
它的作用主要是验证方法在现代数据集上的稳定性和泛化能力。

### 5.3 CIC-IDS2017

这个数据集做了两种设定：

- `cic-ids2017-random`
  - 随机分层划分
  - 更适合做主结果表

- `cic-ids2017`
  - 按工作日划分
  - 更适合做跨天泛化补充实验

## 6. 对比模型

目前已经纳入的对比模型包括：

- `Vanilla PPO`
- `MARL`
- `Bi-ARL`
- `LSTM-IDS`
- `HGBT-IDS`
- `XGBoost-IDS`
- `LightGBM-IDS`

其中后三个表格模型是在后续补充实验中加入的现代强 baseline。

## 7. 主要实验结果

### 7.1 NSL-KDD 上的结果

在 `NSL-KDD` 上，`Bi-ARL` 的结果是：

- `F1 = 80.17%`
- `FPR = 10.41%`

与 `Vanilla PPO` 相比：

- `F1` 有明显提升
- `FPR` 从 `39.57%` 降到了 `10.41%`

这说明在受控 benchmark 上，bi-level 对普通 RL 是有效的。

### 7.2 UNSW-NB15 上的结果

在 `UNSW-NB15` 上，`Bi-ARL` 的均值结果大约是：

- `F1 = 54.13%`
- `FPR = 53.56%`

而强监督表格模型结果更好：

- `HGBT-IDS`：`F1 = 89.45%`
- `LSTM-IDS`：`F1 = 87.30%`

这说明：

- 当前 `Bi-ARL` 在现代数据集上还不稳定
- 还不能和强监督模型正面竞争

### 7.3 CIC-IDS2017-random 上的结果

在随机分层版本上，树模型接近饱和：

- `HGBT-IDS`：`F1 = 99.74%`
- `XGBoost-IDS`：`F1 = 99.74%`
- `LightGBM-IDS`：`F1 = 99.75%`

`LSTM-IDS` 结果为：

- `F1 = 94.93%`

而 RL 的 smoke test 明显更弱：

- `Bi-ARL seed42`：`F1 = 36.48%`
- `Vanilla PPO seed42`：`F1 = 33.17%`

这说明在 `CIC-IDS2017-random` 上，当前 RL 线并不具备竞争力。

### 7.4 CIC-IDS2017 日期划分结果

在严格按工作日划分时，结果明显变差。

例如：

- `XGBoost-IDS`：`F1 = 38.26%`

这说明数据划分协议会极大影响结论。  
因此论文里不能只用一种划分方式就给出过强结论。

## 8. 当前结论

目前可以得出的结论是：

1. `Bi-ARL` 在 `NSL-KDD` 上相对普通 RL 是有效的。
2. `Bi-ARL` 当前最能成立的贡献，是提供了一个有研究意义的 bi-level RL 框架。
3. 在 `UNSW-NB15` 和 `CIC-IDS2017` 上，强监督表格模型明显更强。
4. 因此，这项工作的价值更偏向：
   - 方法框架
   - 负面结果分析
   - 现代数据集上的局限性揭示

而不是“全面领先的最优 IDS 模型”。

## 9. 创新点概括

我认为当前最合理的创新点有三点：

1. 将网络威胁检测建模为一个 bi-level attacker-defender 优化问题
2. 用 RL 的方式显式逼近 attacker 的更强响应，而不是只做普通 simultaneous-update
3. 不只给出正结果，也系统展示了该方法在现代数据集和不同评测协议下的局限性

## 10. 当前不足

主要不足有：

1. RL 在现代数据集上的性能还不够强
2. 种子敏感性仍然存在
3. 还没有纳入 Transformer / SSL / GNN 这类更近期的深度 baseline
4. 目前最强结果仍然集中在 `NSL-KDD`，这会限制投稿层级

## 11. 下一步计划

后续如果继续提升，优先级建议如下：

1. 增加现代强 baseline
   - 优先 Transformer
   - 或补一个公开实现的 SSL/GNN

2. 增加更强的现代数据集
   - 如 `CSE-CIC-IDS2018`

3. 继续优化 RL 方法
   - inner loop 稳定性
   - reward 设计
   - seed 敏感性

## 12. 汇报总结

总结来说，这个项目现在已经不是一个半成品仓库，而是一个具备论文草稿、实验结果和多数据集对比的研究型项目。

但也必须保持清醒：

- `Bi-ARL` 目前在受控 benchmark 上成立
- 在现代数据集上还不够强
- 因此更适合定位成“有研究价值的方法探索”，而不是“已经成熟的高性能 IDS 系统”
