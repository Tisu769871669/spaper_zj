# 路线 C：面向 CCF-A 的重构记录

## 1. 为什么必须重构

当前旧主线 `Bi-ARL (PPO defender)` 的结论只在 `NSL-KDD (2009)` 上相对成立。

在更现代的数据集上：

- `UNSW-NB15 (2015)`：当前 RL 方法明显落后于监督学习强基线
- `CIC-IDS2017 (2017/2018)`：树模型几乎饱和随机划分协议，RL smoke test 明显偏弱

因此，如果目标是 `CCF-A`，论文主线不能继续写成：

- `PPO-based detector is strong on modern IDS benchmarks`

而应改写为：

- `Bi-level adversarial training for supervised intrusion detection`

## 2. 新主线的核心思想

保留原项目的创新点：

- attacker-defender 对抗
- bi-level / leader-follower 训练思想
- worst-case robustness motivation

但把“检测器本体”从 RL 策略网络，切换为更强的监督学习模型。

当前第一版实现为：

- 下层：攻击者通过多步输入扰动，最大化检测器损失
- 上层：MLP 检测器最小化 clean loss + adversarial loss

这相当于一个更适合现代 tabular IDS 的双层对抗训练框架。

## 3. 已完成的第一版实现

代码文件：

- `src/baselines/bilevel_supervised_ids.py`

主要内容：

- `MLPIDS`：监督式 MLP 检测器
- `BiLevelSupervisedTrainer`：双层对抗训练器
- inner loop：迭代生成扰动样本
- outer loop：优化检测器在 clean + adversarial 样本上的分类性能

模型保存路径：

- `outputs/models/<dataset>/BiATMLP/seed<seed>/model.pth`

结果保存路径：

- `outputs/results/biat_mlp_<dataset>_seed<seed>.json`

## 4. 第一批结果信号

### UNSW-NB15（seed=42, 5 epochs）

结果文件：

- `outputs/results/biat_mlp_unsw_nb15_seed42.json`

clean 指标：

- Recall = `94.39%`
- Precision = `81.98%`
- F1 = `87.75%`
- FPR = `25.43%`

说明：

- 这已经明显强于当前旧 RL 主线
- clean F1 已接近 `LSTM-IDS`
- FPR 明显优于当前 `LSTM-IDS` 均值 (`32.26%`)
- 仍落后于 `LightGBM-IDS` / `XGBoost-IDS`

### NSL-KDD（seed=42, 5 epochs）

结果文件：

- `outputs/results/biat_mlp_nsl_kdd_seed42.json`

clean 指标：

- Recall = `66.97%`
- Precision = `97.18%`
- F1 = `79.30%`
- FPR = `2.56%`

说明：

- FPR 非常低
- F1 与原 `Bi-ARL` 接近，但目前还没超过其多种子均值
- 这是一个更合理的“强监督 + 对抗训练”起点

## 5. 当前数据集来源说明

### 5.1 NSL-KDD

- 时间：`2009`
- 来源：UNB CIC 官方页
- 官方链接：<https://www.unb.ca/cic/datasets/nsl.html>
- 引用论文：Tavallaee et al., *A Detailed Analysis of the KDD CUP 99 Data Set*, 2009

这篇工作做了什么：

- 分析 `KDD'99` 的重复样本和分布问题
- 通过清洗与重构形成 `NSL-KDD`
- 让结果更适合公平比较

### 5.2 UNSW-NB15

- 时间：`2015`
- 来源：UNSW 官方数据页
- 官方链接：<https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- 引用论文：Moustafa and Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS 2015

这篇工作做了什么：

- 在 UNSW Cyber Range 中采集更现代的正常流量和攻击流量
- 构造 `49` 个原始流特征
- 提供官方 train/test split

### 5.3 CIC-IDS2017

- 时间：采集场景对应 `2017`，系统介绍论文 `2018`
- 来源：Canadian Institute for Cybersecurity / UNB
- 论文链接：<https://www.scitepress.org/papers/2018/66398/pdf/index.html>
- 引用论文：Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018

这篇工作做了什么：

- 提出新的 IDS 数据集构建标准
- 按标准生成 `CICIDS2017`
- 覆盖 DDoS、Brute Force、Web attacks、PortScan 等现代攻击

### 5.4 CICIoT2023

- 时间：`2023`
- 来源：UNB CIC 官方数据页
- 官方链接：<https://www.unb.ca/cic/datasets/iotdataset-2023.html>
- 引用论文：Neto et al., *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment*, 2023

这篇工作做了什么：

- 在 `105` 个 IoT 设备组成的拓扑中采集流量
- 构造 `33` 种攻击
- 提供面向现代 IoT 场景的大规模基准

## 6. 接下来真正要做的事

1. 补齐 `BiAT-MLP` 在 `UNSW-NB15` 的 4 seeds
2. 接着跑 `CIC-IDS2017-random`
3. 如果趋势成立，再接 `CICIoT2023` 完整官方数据
4. 然后再考虑：
   - 更强检测器：`FT-Transformer`
   - 更强 attacker：feature-group / sparse attacker
   - 更严格鲁棒性协议

## 7. 当前阶段结论

路线 C 已经比旧 RL 主线更有希望。

虽然第一版 `BiAT-MLP` 还没有超过 `LightGBM/XGBoost`，但它已经在 `UNSW-NB15` 上表现出：

- 现代数据集上可训练
- 相比旧 RL 主线明显更强
- 有继续优化并形成正面结果的空间

这说明项目现在终于从“老数据集上的 RL 故事”转向了“现代数据集上的可竞争对抗训练框架”。
