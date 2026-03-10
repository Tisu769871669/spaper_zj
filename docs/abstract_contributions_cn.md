# 中文摘要与创新点

## 一、中文摘要

网络威胁检测面临攻击者自适应、数据分布漂移以及对抗扰动等挑战。本文从双层对抗强化学习出发，首先构建了一个基于 Stackelberg 博弈的 Bi-level Adversarial Reinforcement Learning（Bi-ARL）框架，并在此基础上进一步将相同的双层对抗训练思想迁移到更强的监督式表格检测器中，形成路线 C。本文在 NSL-KDD、UNSW-NB15 和 CIC-IDS2017 三个公开数据集上进行了系统实验。结果表明：在 NSL-KDD 上，Bi-ARL 在强化学习方法中取得了最高的平均 F1 值 80.17%，并将 FPR 从 Vanilla PPO 的 39.57% 降低到 10.41%；在更现代的 UNSW-NB15 上，路线 C 中的 BiAT-FTTransformer 取得了 89.51% 的平均 F1 和 14.97% 的平均 FPR，成为本文中最强的神经网络检测器，并在该数据集上取得了最低的误报率；在 CIC-IDS2017 随机分层协议下，迁移最优 UNSW 配置后，BiAT-FTTransformer 进一步提升到 87.09% 的平均 F1 和 6.78% 的平均 FPR，但仍明显落后于 LightGBM、XGBoost 等强表格基线。综合来看，显式的双层对抗训练在受控基准和现代数据集上均具有方法价值，FT-Transformer 骨干可以显著增强现代数据表现，但整体框架仍未达到最强 boosted-tree 基线的水平。

## 二、创新点

### 创新点 1：将网络威胁检测建模为双层 attacker-defender 优化问题

本文不是把 IDS 仅仅看成一个普通分类任务，而是将其建模为 attacker 和 defender 的双层优化过程。该建模方式更贴近“攻击者不断适应、检测器需要持续鲁棒化”的安全场景。

### 创新点 2：从纯 RL 检测器扩展到监督式双层对抗训练框架

本文没有停留在最初的 PPO 检测器实现上，而是将相同的双层对抗思想迁移到监督式表格模型中，形成路线 C，并实现了：

- `BiAT-MLP`
- `BiAT-FTTransformer`

这使得方法创新点从“某个特定 RL 检测器”提升为“可迁移到不同 detector backbone 的双层对抗训练框架”。

### 创新点 3：在现代公开数据集上验证方法的有效性与边界

本文不仅在 `NSL-KDD` 这类传统基准上给出正结果，还在 `UNSW-NB15` 与 `CIC-IDS2017` 上进行了系统验证，明确区分：

- 哪些结论在受控 benchmark 上成立
- 哪些结论在现代数据集上仍然成立
- 哪些地方仍然落后于强基线

这种“既给正结果，也给边界条件”的实验结构，本身就是论文可信度的重要部分。

## 三、当前最能站住的贡献表述

如果要写进论文摘要、创新点或答辩材料，当前最稳妥的表达是：

1. 提出并实现了一个可扩展的双层对抗训练框架，用于网络威胁检测。
2. 在 `NSL-KDD` 上验证了 Bi-ARL 对普通 RL 的有效性。
3. 在 `UNSW-NB15` 上，基于 FT-Transformer 的路线 C 取得了 89.51% 的平均 F1 和 14.97% 的平均 FPR，成为当前最强的神经网络方案。
4. 在 `CIC-IDS2017-random` 上，最优 `UNSW` 配置迁移后仍保持小幅正增益，说明该优化具有一定迁移性，但整体仍落后于最强表格基线。

## 四、当前不建议夸大的表述

以下说法当前都不建议使用：

- state of the art
- 全面优于现代 IDS baseline
- 在现代公开数据集上全面领先
- 已经解决跨数据集鲁棒性问题

更合适的写法是：

- strongest neural model in our study
- competitive bilevel adversarial training framework
- lower false positive rate on UNSW-NB15
- improved modern-data performance, but still below boosted-tree baselines on overall F1
