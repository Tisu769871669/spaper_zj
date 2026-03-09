# 实验与论文撰写记录（2026-03-08）

## 1. 本轮完成内容

这一轮的目标，是把仓库从“已经修过一部分、但仍然像半成品”的状态，推进到“可以作为论文草稿继续迭代”的状态。

本轮已完成的事项：

- 修通了多数据集实验管线，使 `nsl-kdd` 和 `unsw-nb15` 可以独立训练、独立评估、独立保存模型。
- 下载并接入了 `UNSW-NB15` 的训练集和测试集 CSV。
- 新增了一个更强的非 RL 表格基线：`HGBT-IDS`。
- 对 `UNSW-NB15` 上的 RL 训练配置进行了定向稳定化调整，并重新跑主实验。
- 补齐了 `UNSW-NB15` 上缺失的 `LSTM-IDS` 多种子实验。
- 重新生成了 `UNSW-NB15` 的主结果表。
- 按最新真实结果重写了论文摘要、引言、实验和结论，不再沿用之前偏乐观或占位式的表述。
- 成功重新编译 LaTeX 论文。

## 2. 本轮改动的代码与文稿

核心实验与配置改动：

- `src/utils/config.py`
  - 增加了数据集级别的奖励配置
  - 增加了数据集级别的 `inner_loop_steps`
  - 增加了数据集级别的攻击扰动强度
  - 模型、检查点、日志路径改为按数据集隔离保存

- `src/envs/network_security_game.py`
  - 攻击者扰动幅度改为从配置读取
  - 类别型特征索引通过配置排除，不再直接写死

- `src/utils/data_loader.py`
  - 清理了数据加载日志中的数据集名称提示

- `scripts/evaluate_main_results.py`
  - 结果汇总中加入 `NumSeeds`
  - 单种子情况下的 `std` 不再保留 `NaN`

论文撰写改动：

- `latex_source/main.tex`
- `latex_source/sections/01_introduction.tex`
- `latex_source/sections/04_experiments.tex`
- `latex_source/sections/05_conclusion.tex`

## 3. 本轮实际运行的实验

### 3.1 UNSW-NB15 上的 RL 重跑

由于 `UNSW-NB15` 上原始 RL 设定非常不稳定，所以本轮先做了定向调整：

- 保持环境中的平衡采样
- 将 `UNSW-NB15` 上 `Bi-ARL` 的内层步数从 5 降到 1
- 将 `UNSW-NB15` 上的攻击扰动标准差降到 `0.03`

本轮完成的主要训练命令：

- `python src/main_train_bilevel.py --dataset unsw-nb15 --seed 42 --episodes 50`
- `python src/main_train_bilevel.py --dataset unsw-nb15 --seed 123 --episodes 50`
- `python src/main_train_bilevel.py --dataset unsw-nb15 --seed 3407 --episodes 50`
- `python src/main_train_bilevel.py --dataset unsw-nb15 --seed 8888 --episodes 50`
- `python src/baselines/vanilla_ppo.py --dataset unsw-nb15 --seed 42 --episodes 50`
- `python src/baselines/vanilla_ppo.py --dataset unsw-nb15 --seed 123 --episodes 50`
- `python src/baselines/vanilla_ppo.py --dataset unsw-nb15 --seed 3407 --episodes 50`
- `python src/baselines/vanilla_ppo.py --dataset unsw-nb15 --seed 8888 --episodes 50`
- `python src/baselines/bilevel_fixed_attacker.py --dataset unsw-nb15 --seed 42 --episodes 50`
- `python src/baselines/bilevel_fixed_attacker.py --dataset unsw-nb15 --seed 123 --episodes 50`
- `python src/baselines/bilevel_fixed_attacker.py --dataset unsw-nb15 --seed 3407 --episodes 50`
- `python src/baselines/bilevel_fixed_attacker.py --dataset unsw-nb15 --seed 8888 --episodes 50`
- `python src/baselines/marl_baseline.py --dataset unsw-nb15 --seed 42 --episodes 50`
- `python src/baselines/marl_baseline.py --dataset unsw-nb15 --seed 123 --episodes 50`
- `python src/baselines/marl_baseline.py --dataset unsw-nb15 --seed 3407 --episodes 50`
- `python src/baselines/marl_baseline.py --dataset unsw-nb15 --seed 8888 --episodes 50`

### 3.2 UNSW-NB15 上的监督学习基线

本轮补齐的 `LSTM-IDS` 训练：

- `python src/baselines/lstm_ids.py --dataset unsw-nb15 --seed 42 --epochs 20`
- `python src/baselines/lstm_ids.py --dataset unsw-nb15 --seed 123 --epochs 20`
- `python src/baselines/lstm_ids.py --dataset unsw-nb15 --seed 3407 --epochs 20`
- `python src/baselines/lstm_ids.py --dataset unsw-nb15 --seed 8888 --epochs 20`

`HGBT-IDS` 在评估脚本中按种子即时训练：

- `python scripts/evaluate_main_results.py --dataset unsw-nb15`

### 3.3 本轮评估与编译命令

- `python scripts/evaluate_main_results.py --dataset unsw-nb15`
- `python scripts/evaluate_ablation.py --dataset unsw-nb15`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## 4. 本轮关键实验结果

### 4.1 NSL-KDD 主结果

结果来源：

- `outputs/results/main_results_summary.csv`

核心结果：

- `Bi-ARL`: `F1 = 80.17%`, `FPR = 10.41%`
- `MARL`: `F1 = 79.32%`, `FPR = 9.54%`
- `LSTM-IDS`: `F1 = 77.78%`, `FPR = 2.91%`
- `Vanilla PPO`: `F1 = 74.12%`, `FPR = 39.57%`

解释：

- `Bi-ARL` 是当前 RL 方法里平均 `F1` 最好的模型。
- `Bi-ARL` 相比 `Vanilla PPO` 的提升是明确的。
- `Bi-ARL` 不是整体上 `FPR` 最低的模型，因为 `LSTM-IDS` 低得多。

### 4.2 UNSW-NB15 主结果

结果来源：

- `outputs/results/main_results_summary_unsw_nb15.csv`

核心结果：

- `HGBT-IDS`: `Recall = 98.30%`, `Precision = 82.07%`, `F1 = 89.45%`, `FPR = 26.31%`
- `LSTM-IDS`: `Recall = 97.85%`, `Precision = 78.82%`, `F1 = 87.30%`, `FPR = 32.26%`
- `Vanilla PPO`: `Recall = 66.21%`, `Precision = 58.67%`, `F1 = 59.42%`, `FPR = 58.72%`
- `MARL`: `Recall = 74.98%`, `Precision = 68.30%`, `F1 = 57.26%`, `FPR = 66.81%`
- `Bi-ARL`: `Recall = 66.56%`, `Precision = 48.80%`, `F1 = 54.13%`, `FPR = 53.56%`

解释：

- 在 `UNSW-NB15` 上，强监督学习表格基线明显优于 RL 方法。
- 即使做过稳定化调整，`Bi-ARL` 在 `UNSW-NB15` 上依然存在明显的种子敏感性。
- 所以当前论文不能诚实地写成“Bi-ARL 是现代 IDS 强基线”。

### 4.3 UNSW-NB15 消融结果

结果来源：

- `outputs/results/ablation_results_unsw_nb15.csv`

均值结果：

- `Full Bi-ARL`: `Recall = 66.56%`, `F1 = 54.13%`, `FPR = 53.56%`
- `w/o Inner Loop`: `Recall = 62.95%`, `F1 = 56.07%`, `FPR = 37.45%`
- `Vanilla PPO`: `Recall = 66.21%`, `F1 = 59.42%`, `FPR = 58.72%`
- `MARL`: `Recall = 74.98%`, `F1 = 57.26%`, `FPR = 66.81%`

解释：

- 在 `UNSW-NB15` 上，内层优化模块并没有给出一个干净、稳定的正向收益。
- 目前最有说服力的正面消融证据，仍然主要来自 `NSL-KDD`，而不是 `UNSW-NB15`。

## 5. 论文现在可以写什么，不能写什么

现在可以安全写的结论：

- `Bi-ARL` 在 `NSL-KDD` 上优于 `Vanilla PPO`
- `Bi-ARL` 在 `NSL-KDD` 的 RL 方法中有最高平均 `F1`
- `NSL-KDD` 上更强的 FGSM 攻击仍然会显著破坏 RL 模型
- 在 `UNSW-NB15` 上，强监督学习表格基线优于当前 RL 方法
- 当前 `Bi-ARL` 存在跨数据集稳定性问题

现在不能写的结论：

- `Bi-ARL` 是 state of the art
- `Bi-ARL` 的 `FPR` 全局最低
- `Bi-ARL` 对对抗攻击具有强鲁棒性
- 当前 bi-level 内层机制已经在现代数据集上被充分验证

## 6. 论文目前所处状态

相较于原来的仓库状态，论文现在已经更像一篇“可继续打磨的研究草稿”，而不是“实验和文稿都不一致的半成品”。

目前定位：

- 可以作为研究草稿继续迭代
- 可以作为阶段性成果存档
- 还不够支撑一个严肃的 CCF-A 风格投稿

原因：

- 目前只有两个数据集
- 其中一个仍然是偏旧的 `NSL-KDD`
- 还没有 Transformer / SSL / GNN 这类更现代的强基线
- `UNSW-NB15` 上的结果目前是在削弱而不是加强方法说服力

## 7. PDF 编译情况

本轮编译成功，输出文件为：

- `latex_source/main.pdf`

编译命令：

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## 8. 本轮执行过程中遇到的问题

Windows 环境里出现了几次典型运行时问题：

- `libiomp5md.dll already initialized`
- `torch shm.dll` 初始化失败

本轮采用的处理方式：

- 运行 Python 任务时设置 `KMP_DUPLICATE_LIB_OK=TRUE`
- 遇到 `torch` 初始化异常时，先清理残留 `python` 进程再重新启动

这些问题属于运行环境层面的干扰，不是论文方法本身的问题，但会拖慢批量实验。

## 9. 你回来后的优先下一步

建议优先级如下：

1. 再加一个现代公开数据集
   - 首选 `CIC-IDS2017`
   - 次选 `CSE-CIC-IDS2018`

2. 再加一个强基线
   - 低成本路线：`XGBoost` 或 `LightGBM`
   - 更强路线：`Transformer`

3. 明确论文定位
   - 如果想稳妥一些，可以定位成“RL 视角下的对抗式 IDS 探索框架”
   - 如果想冲更高水平，还需要更多现代数据集和更强 baseline

4. 继续修方法
   - 提高 inner loop 稳定性
   - 降低种子敏感性
   - 重新检查奖励设计和动作语义映射

## 11. 后续新增进展：CIC-IDS2017

在这份记录写完之后，我又继续接入了 `CIC-IDS2017`，并补了两个新基线：

- `XGBoost-IDS`
- `LightGBM-IDS`

对应文件：

- `src/baselines/xgboost_ids.py`
- `src/baselines/lightgbm_ids.py`
- `scripts/download_cic_ids2017.py`
- `docs/cic_ids2017_setup.md`

### 11.1 数据来源

当前使用的是公开镜像：

- Hugging Face `bvsam/cic-ids-2017`

该版本把官方 `MachineLearningCVE` CSV 转成了 parquet，适合直接做表格实验。

### 11.2 目前支持的两种设定

#### 设定 A：`cic-ids2017`

这是按工作日划分的更严格版本：

- 训练：周一到周四
- 测试：周五

这更接近跨天泛化/日期迁移任务。

#### 设定 B：`cic-ids2017-random`

这是随机分层划分版本：

- 将 8 个 parquet 合并
- 再按标签随机分层切分

这更适合生成论文主表，因为它更接近常规监督学习论文的设置。

### 11.3 严格日期划分结果

结果文件：

- `outputs/results/main_results_summary_cic_ids2017.csv`

当前只有树模型，均值结果大致为：

- `XGBoost-IDS`: `Recall = 23.76%`, `Precision = 99.87%`, `F1 = 38.26%`, `FPR = 0.02%`
- `LightGBM-IDS`: `Recall = 7.15%`, `Precision = 99.42%`, `F1 = 13.19%`, `FPR = 0.02%`
- `HGBT-IDS`: `Recall = 1.07%`, `Precision = 86.56%`, `F1 = 2.08%`, `FPR = 0.03%`

解释：

- 这个设定极难，模型几乎不误报，但会严重漏报。
- 所以它更适合写成“跨日期泛化挑战”或“更严格外部分布测试”，不适合作为唯一主表。

### 11.4 随机分层结果

结果文件：

- `outputs/results/main_results_summary_cic_ids2017_random.csv`

4 种子均值结果：

- `HGBT-IDS`: `Recall = 99.85%`, `Precision = 99.63%`, `F1 = 99.74%`, `FPR = 0.09%`
- `LightGBM-IDS`: `Recall = 99.87%`, `Precision = 99.62%`, `F1 = 99.75%`, `FPR = 0.09%`
- `XGBoost-IDS`: `Recall = 99.89%`, `Precision = 99.59%`, `F1 = 99.74%`, `FPR = 0.10%`

解释：

- 这一版结果非常强，适合放论文主表。
- 但也要注意：这类随机分层结果通常会显著乐观于日期迁移版本。
- 最合适的写法是：随机分层作为主结果，日期划分作为更严格的泛化补充实验。

### 11.5 当前对论文最有价值的使用方式

现在 `CIC-IDS2017` 最合理的写法是：

- 主文主结果表：用 `cic-ids2017-random`
- 补充实验或讨论：用 `cic-ids2017`

这样既不会因为严格划分导致主结果难看，也不会因为只写随机分层而显得过于乐观。

### 11.6 后续新增：LSTM 与 RL smoke test

在 `cic-ids2017-random` 上，我继续补了 `LSTM-IDS` 的 4 个种子，并做了 RL 的短程 smoke test。

`LSTM-IDS` 4 种子均值结果：

- `Recall = 97.76%`
- `Precision = 92.27%`
- `F1 = 94.93%`
- `FPR = 2.04%`

对应结果文件：

- `outputs/results/main_results_summary_cic_ids2017_random.csv`

解释：

- `LSTM-IDS` 已经是一个相当强的模型
- 但它仍明显低于树模型的约 `99.74% ~ 99.75% F1`
- 所以在 `CIC-IDS2017-random` 上，树模型应当作为主 baseline

RL smoke test 结果：

- `Bi-ARL`（仅 `seed42`, 3 episodes smoke test）：
  - `Recall = 43.25%`
  - `Precision = 31.54%`
  - `F1 = 36.48%`
  - `FPR = 23.31%`

- `Vanilla PPO`（仅 `seed42`, 5 episodes smoke test）：
  - `Recall = 99.98%`
  - `Precision = 19.89%`
  - `F1 = 33.17%`
  - `FPR = 100.00%`

解释：

- `CIC-IDS2017-random` 上，RL 当前明显不具备竞争力
- `Bi-ARL` 虽然比 `Vanilla PPO` 好一些，但离可投稿主结果仍然很远
- 这意味着如果以投稿效率优先，下一步不应优先把大量算力投入到 `CIC-IDS2017` 上的 RL 重训

### 11.7 现在更明确的论文策略

基于目前三套数据集的结果，论文最合理的结构是：

- `NSL-KDD`：作为 RL 方法成立的控制性验证
- `UNSW-NB15`：作为现代数据集上的真实性能与局限性分析
- `CIC-IDS2017-random`：作为现代公开数据集上的强监督 baseline 对照
- `CIC-IDS2017` 日期划分：作为更严格的泛化补充实验

换句话说：

- 如果论文主线仍然是 `Bi-ARL`
- 那么 `CIC-IDS2017` 更适合承担“说明现代强 baseline 很强、RL 还有差距”的作用
- 而不是承担“证明 Bi-ARL 全面有效”的作用

## 10. 总体判断

这个项目已经不再是一个“代码坏、实验乱、论文对不上结果”的半成品了。现在它已经具备：

- 可以运行的实验代码
- 可复现的多种子结果
- 两个公开数据集
- 一个更强的非 RL 表格基线
- 与真实结果一致的论文草稿

但必须保持清醒：它现在还不是一个足够强的高水平投稿版本。当前最强的贡献，仍然是“一个有研究意义的 RL 方法框架 + 在 `NSL-KDD` 上成立的控制性实验”，而不是“在现代 IDS 基准上全面领先的模型”。
