# 答辩/PPT 提纲

## 1. 封面

- 题目：基于双层对抗训练的网络威胁检测方法研究
- 作者、学院、导师、日期

## 2. 研究背景

- 网络威胁检测面临的三个现实问题：
  - 攻击者可自适应
  - 数据分布会变化
  - 传统 benchmark 结果不代表现代数据集表现
- 引出核心问题：
  - 能否把 IDS 训练成 attacker-defender 的双层优化过程

## 3. 研究目标

- 最初目标：提出 `Bi-ARL`
- 演化目标：保留 bi-level 思想，同时提升现代数据集表现
- 最终形成两条线：
  - 旧主线：`Bi-ARL`
  - 新主线：`BiAT-MLP / BiAT-FTTransformer`

## 4. 方法总体框架

- attacker / defender 交互
- inner loop / outer loop
- 旧主线：PPO detector
- 新主线：监督式 tabular detector
- 强调：创新点在于 bilevel adversarial training，而不是单一 backbone

## 5. 数据集与来源

建议这一页按“时间 + 论文 + 作用”讲：

- `NSL-KDD (2009)`
  - Tavallaee et al.
  - 作用：受控 benchmark，验证旧 RL 主线
- `UNSW-NB15 (2015)`
  - Moustafa and Slay
  - 作用：现代通用主实验
- `CIC-IDS2017 (2017/2018)`
  - Sharafaldin et al.
  - 作用：第二个现代公开基准

## 6. 对比模型

- RL 系：
  - `Vanilla PPO`
  - `MARL`
  - `Bi-ARL`
- 神经网络系：
  - `LSTM-IDS`
  - `BiAT-MLP`
  - `BiAT-FTTransformer`
- 强表格系：
  - `HGBT-IDS`
  - `XGBoost-IDS`
  - `LightGBM-IDS`

## 7. NSL-KDD 结果

建议核心信息只保留一句：

- `Bi-ARL` 在 RL 方法里最好：
  - `F1 = 80.17%`
  - `FPR = 10.41%`
- 对比 `Vanilla PPO`：
  - `FPR 39.57% -> 10.41%`

结论：

- 旧主线在受控 benchmark 上成立

## 8. UNSW-NB15 结果

这是全篇最重要的一页。

建议突出：

- `BiAT-FTTransformer`
  - `F1 = 89.51%`
  - `FPR = 14.97%`
- `BiAT-MLP`
  - `F1 = 86.40%`
  - `FPR = 28.31%`
- `LightGBM`
  - `F1 = 91.86%`
  - `FPR = 17.14%`

结论：

- 路线 C 显著优于旧 RL 线
- `FTTransformer` 是当前最强神经网络方案
- 在 `UNSW-NB15` 上拿到了所有对照方法中最低的 `FPR`
- 但总体 `F1` 仍未超过最强树模型

## 9. CIC-IDS2017-random 结果

建议突出：

- 树模型几乎饱和：
  - `LightGBM ≈ 99.75% F1`
- `BiAT-FTTransformer`
  - 迁移优化后 `F1 = 87.09%`
  - `FPR = 6.78%`

结论：

- 迁移配置后仍有小幅正增益
- 但该数据集上路线 C 仍明显落后于树模型

## 10. 主要结论

- `Bi-ARL` 在 `NSL-KDD` 上成立
- 路线 C 说明：同样的 bi-level 思想迁移到更强 detector 后，现代数据表现明显改善
- `BiAT-FTTransformer` 是当前主方法
- 论文最稳的正结果来自：
  - `UNSW-NB15`
  - `89.51% F1`
  - `14.97% FPR`

## 11. 创新点

- 将 IDS 训练建模为双层 attacker-defender 优化
- 将 bilevel adversarial training 从 RL 检测器迁移到监督式 tabular detector
- 在多个现代公开数据集上验证方法的有效性与边界

## 12. 不足与局限

- 旧 RL 主线在现代数据集上弱
- `CIC-IDS2017-random` 上仍明显落后于最强表格基线
- 方法总体仍不能写成 SOTA

## 13. 下一步工作

- 补更现代或更严格协议的数据集
- 增加 SSL / GNN 等近期强基线
- 进一步稳定 inner loop
- 继续提升现代数据集上的总体 F1

## 14. 汇报结束页

- 感谢聆听
- 欢迎老师提问
