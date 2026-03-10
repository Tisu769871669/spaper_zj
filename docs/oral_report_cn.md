# 小论文中文汇报稿

## 1. 课题题目

基于双层对抗训练的网络威胁检测方法研究

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

本课题最初的目标是提出一个 Bi-level Adversarial Reinforcement Learning，也就是 `Bi-ARL` 框架，用于网络威胁检测。

核心思想是：

- defender 作为 leader
- attacker 作为 follower
- 训练时不是简单同时更新两个 agent
- 而是先让 attacker 朝着更强响应逼近，再让 defender 针对该 attacker 更新

随着实验推进，我们进一步把这套 bi-level 思想迁移到强监督检测器上，形成了路线 C，也就是：

- `BiAT-MLP`
- `BiAT-FTTransformer`

这样做的目的，是希望 defender 不仅适应普通样本，还能针对更强扰动进行训练，同时避免纯 RL 检测器在现代数据集上的性能瓶颈。

## 4. 方法简介

方法整体现在分成两条线：

- 旧主线：`Bi-ARL`
- 新主线：`BiAT`

整体包括四部分：

1. 网络安全对抗环境
   - 将流量样本作为状态
   - attacker 对攻击流量施加扰动
   - defender 根据扰动后的观测做判定

2. 攻防训练机制
   - 旧主线中 attacker 和 defender 都基于 PPO
   - 新主线中 defender 改为监督式 tabular detector

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

在 `UNSW-NB15` 上，旧 `Bi-ARL` 的均值结果大约是：

- `F1 = 54.13%`
- `FPR = 53.56%`

而强监督表格模型结果更好：

- `BiAT-MLP`：`F1 = 86.40%`
- `BiAT-FTTransformer`：`F1 = 89.51%`
- `LightGBM-IDS`：`F1 = 91.86%`

这说明：

- 当前 `Bi-ARL` 在现代数据集上还不稳定
- 但路线 C 已经取得了明显更强的现代数据结果
- 其中 `BiAT-FTTransformer` 的 `FPR = 14.97%`，已经是当前对照方法里最低的

### 7.3 CIC-IDS2017-random 上的结果

在随机分层版本上，树模型接近饱和：

- `HGBT-IDS`：`F1 = 99.74%`
- `XGBoost-IDS`：`F1 = 99.74%`
- `LightGBM-IDS`：`F1 = 99.75%`

`LSTM-IDS` 结果为：

- `F1 = 94.98%`

而 RL 的 smoke test 明显更弱：

- `Bi-ARL seed42`：`F1 = 36.48%`
- `Vanilla PPO seed42`：`F1 = 33.17%`

进一步把 `UNSW` 上最优 `FT` 配置迁移过来后，`BiAT-FTTransformer` 提升到：

- `F1 = 87.09%`
- `FPR = 6.78%`

这说明在 `CIC-IDS2017-random` 上：

- 路线 C 明显强于旧 RL 线
- 最优 `UNSW` 配置具备一定迁移性
- 但收益仍然有限，远未接近树模型

### 7.4 CIC-IDS2017 日期划分结果

在严格按工作日划分时，结果明显变差。

例如：

- `XGBoost-IDS`：`F1 = 38.26%`

这说明数据划分协议会极大影响结论。  
因此论文里不能只用一种划分方式就给出过强结论。

## 8. 当前结论

目前可以得出的结论是：

1. `Bi-ARL` 在 `NSL-KDD` 上相对普通 RL 是有效的。
2. 路线 C 说明：同样的 bi-level 思想迁移到更强监督检测器后，现代数据表现会明显改善。
3. `BiAT-FTTransformer` 已经成为当前最强神经网络主方法。
4. 在 `UNSW-NB15` 上，它拿到了：
   - `F1 = 89.51%`
   - `FPR = 14.97%`
5. 在 `CIC-IDS2017-random` 上，它仍明显落后于树模型，因此论文不能写成 SOTA。

而不是“全面领先的最优 IDS 模型”。

## 9. 创新点概括

我认为当前最合理的创新点有三点：

1. 将网络威胁检测建模为一个 bi-level attacker-defender 优化问题
2. 将同一套 bi-level 对抗训练思想从纯 RL 检测器扩展到监督式 tabular detector
3. 在现代数据集上实现了一个更强的 `BiAT-FTTransformer` 版本，并验证了其跨数据集的有限迁移性

## 10. 当前不足

主要不足有：

1. 旧 RL 主线在现代数据集上的性能仍然偏弱
2. 当前最强结果主要来自 `UNSW-NB15`
3. 相比 `LightGBM / XGBoost`，整体 `F1` 仍有差距
4. 还没有纳入更近期的 SSL / GNN baseline

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

- `Bi-ARL` 在受控 benchmark 上成立
- 路线 C 已经把现代数据结果明显往前推进
- 但这项工作目前更适合定位成“有竞争力的 bi-level 对抗训练框架”，而不是“已经全面领先的 IDS 系统”
