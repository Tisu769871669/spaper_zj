# Bi-ARL / BiAT 项目完整工作流

本文档记录项目的**完整流程**，重点回答三个问题：

1. 这个项目现在的真实主线是什么
2. 从数据准备到服务器实验、再到论文写作，整体是怎么跑通的
3. 当前哪些内容属于主结果，哪些只是补充或历史遗留

本文档与 [optimization_decision_log.md](/d:/Study/研2/spaper_zj/docs/optimization_decision_log.md) 分工如下：

- `BI_ARL_WORKFLOW.md`
  - 记录完整流程、标准步骤、当前项目结构和推荐执行顺序
- `optimization_decision_log.md`
  - 记录中间遇到的问题、判断、修复思路、失败案例和路线调整原因

---

## 1. 项目当前定位

这个项目已经不是最初单一的 “Bi-ARL on NSL-KDD” 原型，而是分成了两条线：

### 1.1 旧主线：Bi-ARL（RL 检测器）

- 核心思想：
  - attacker / defender 形成双层博弈
  - 使用 PPO 训练 defender
- 主要代码：
  - [main_train_bilevel.py](/d:/Study/研2/spaper_zj/src/main_train_bilevel.py)
  - [bilevel_trainer.py](/d:/Study/研2/spaper_zj/src/algorithms/bilevel_trainer.py)
  - [network_security_game.py](/d:/Study/研2/spaper_zj/src/envs/network_security_game.py)
- 当前结论：
  - 在 `NSL-KDD` 上相对 `Vanilla PPO` 有正面结果
  - 在 `UNSW-NB15`、`CIC-IDS2017` 上不能支撑高水平主结论

### 1.2 新主线：BiAT（强监督检测器 + 双层对抗训练）

- 核心思想：
  - 保留 bi-level / adversarial training 思路
  - 不再强行把检测器限定为 PPO
  - 改成强监督 tabular detector + adversarial inner loop
- 当前实现：
  - `BiAT-MLP`
    - [bilevel_supervised_ids.py](/d:/Study/研2/spaper_zj/src/baselines/bilevel_supervised_ids.py)
  - `BiAT-FTTransformer`
    - [bilevel_fttransformer_ids.py](/d:/Study/研2/spaper_zj/src/baselines/bilevel_fttransformer_ids.py)
- 当前角色：
  - 这是现在更接近 `CCF-A` 目标的主线

---

## 1.3 当前实验进度总表

下表用于快速判断“哪些已经完成，哪些还在进行中，哪些只是补充验证”。

| 模块 | 数据集 | 状态 | 当前结论 | 用途 |
|---|---|---|---|---|
| `Bi-ARL / Vanilla PPO / MARL / FixedAttacker` | `NSL-KDD` | 已完成 | 旧 RL 线在受控基准上成立 | 历史主线、消融、鲁棒性 |
| `Bi-ARL / Vanilla PPO / MARL` | `UNSW-NB15` | 已完成 | 旧 RL 线效果明显不足 | 反证旧路线局限 |
| `BiAT-MLP` | `UNSW-NB15` | 已完成 | 相比旧 RL 线显著提升 | 新主线主结果 |
| `BiAT-FTTransformer` | `UNSW-NB15` | 已完成 | 明显优于 `BiAT-MLP` 和 `LSTM`，但仍低于树模型 | 当前主方法 |
| `XGBoost / LightGBM / LSTM / HGBT` | `UNSW-NB15` | 已完成 | 构成现代强基线 | 主对照组 |
| `BiAT-MLP` | `CIC-IDS2017-random` | 已完成 | 明显优于旧 RL，但仍落后树模型 | 新主线第二主实验 |
| `BiAT-FTTransformer` | `CIC-IDS2017-random` | 已完成 | 比 `BiAT-MLP` 小幅更优，但仍显著落后于树模型和 `LSTM` | backbone 升级验证 |
| `XGBoost / LightGBM / LSTM / HGBT` | `CIC-IDS2017-random` | 已完成 | 树模型非常强 | 主对照组 |
| `XGBoost / LightGBM / BiAT-*` | `CICIoT2023 / grouped` | 已完成 | 当前二分类任务过于容易 | 补充数据集 |
| Windows 服务器批量实验脚本 | `UNSW + CIC` | 已完成 | `core` 已成功跑完，主结果已带回本地 | 正式多种子训练 |

状态解释：

- `已完成`
  - 本地结果和汇总文件已经形成，可直接用于论文或分析
- `进行中`
  - 已经跑通链路，但还在等待服务器正式结果
- `部分完成`
  - 已有单种子或 smoke test 结果，但还不是最终多种子统计

---

## 2. 数据集工作流

### 2.1 当前使用的数据集

| 数据集 | 时间 | 角色 | 当前定位 |
|---|---:|---|---|
| `NSL-KDD` | 2009 | 旧基准 | 受控验证、旧 RL 主线支撑 |
| `UNSW-NB15` | 2015 | 现代通用基准 | 新主线核心主实验 |
| `CIC-IDS2017` | 2017/2018 | 现代公开基准 | 新主线核心主实验 |
| `CICIoT2023` | 2023 | 新 IoT 数据集 | 补充实验，不作为主证明集 |

### 2.2 数据目录

当前数据默认放在：

```text
data/
├─ KDDTrain+.txt
├─ KDDTest+.txt
├─ UNSW_NB15_training-set.csv
├─ UNSW_NB15_testing-set.csv
├─ CIC_IDS2017_machine_learning/
└─ CICIoT2023/
```

### 2.3 数据加载统一入口

统一由：

- [config.py](/d:/Study/研2/spaper_zj/src/utils/config.py)
- [data_loader.py](/d:/Study/研2/spaper_zj/src/utils/data_loader.py)

负责：

- 数据集切换
- 特征预处理
- train/test 划分
- 采样规模控制
- 输出路径隔离

### 2.4 当前支持的数据集键

```text
nsl-kdd
unsw-nb15
cic-ids2017
cic-ids2017-random
ciciot2023
ciciot2023-grouped
```

说明：

- `cic-ids2017`
  - 更严格的按工作日划分
- `cic-ids2017-random`
  - 更适合主结果表的随机分层划分
- `ciciot2023-grouped`
  - 比随机划分更严格，但当前二分类任务仍然偏容易

---

## 3. 模型与实验主线

### 3.1 旧 RL 线

包括：

- `BiARL`
- `VanillaPPO`
- `MARL`
- `FixedAttacker`

用途：

- 支撑旧论文版本
- 作为路线转向前的对照
- 现在主要保留用于：
  - `NSL-KDD`
  - 消融
  - 鲁棒性附加分析

### 3.2 监督 / 表格基线

包括：

- `LSTM-IDS`
- `HGBT-IDS`
- `XGBoost-IDS`
- `LightGBM-IDS`

用途：

- 现代数据集的强对照
- 检验 “纯 RL 检测器是否足够强”
- 为路线 C 提供性能基线

### 3.3 路线 C 主模型

#### BiAT-MLP

- 外层：
  - MLP 检测器
- 内层：
  - 对输入做多步 adversarial perturbation
- 目标：
  - 在 `clean + adversarial` 样本上联合优化

#### BiAT-FTTransformer

- 外层：
  - FT-Transformer 风格 tabular backbone
- 内层：
  - 与 BiAT-MLP 相同的 adversarial inner loop
- 目标：
  - 验证更强 tabular backbone 是否能缩小与树模型的差距

---

## 4. 本地开发工作流

### 4.1 本地机器角色

本地 `4060` 机器负责：

- 代码开发
- 数据接口验证
- 冒烟测试
- 小规模训练
- 论文绘图
- 文档维护

### 4.2 本地环境原则

- `torch` 模型优先使用 GPU 环境
- 当前默认推荐环境：
  - `spaper`

### 4.3 本地常做的事

1. 修改代码
2. 跑单模型 smoke test
3. 跑单数据集 seed=42 验证趋势
4. 生成汇总 CSV
5. 画图
6. 更新论文和文档

---

## 5. 服务器实验工作流

### 5.1 服务器角色

服务器负责：

- 多种子正式训练
- 更长 epoch
- 多数据集批量实验
- 结果统一打包回传

### 5.2 当前服务器环境

已确认可用：

- Windows + conda
- GPU：
  - `Tesla V100-PCIE-32GB`

### 5.3 服务器代码与数据同步

代码走 git：

```powershell
git pull origin main
```

数据不走 git，单独传输：

- ToDesk 文件传输
- 或 PowerShell 同步脚本

相关文档：

- [server_conda_setup.md](/d:/Study/研2/spaper_zj/docs/server_conda_setup.md)

### 5.4 服务器批量运行入口

Windows 服务器当前统一入口：

- [Run-ServerSuite.ps1](/d:/Study/研2/spaper_zj/scripts/Run-ServerSuite.ps1)

常用命令：

```powershell
conda activate spaper
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite core
```

说明：

- `core`
  - 先跑 `UNSW-NB15 + CIC-IDS2017-random`
- `all`
  - 在 `core` 基础上补 `NSL-KDD` 和 `CICIoT2023`

### 5.4.1 双卡并行优化入口

如果目标是继续优化 `UNSW-NB15` 上的 `BiAT-FTTransformer`，当前推荐不再重复跑整套 `core`，而是直接使用双卡并行脚本：

- [Run-DualGpuOptimization.ps1](/d:/Study/研2/spaper_zj/scripts/Run-DualGpuOptimization.ps1)

作用：

- 将不同种子分配到两张 GPU 上并行训练
- 优先服务于：
  - `BiAT-FTTransformer`
  - `BiAT-MLP`
- 当前最推荐场景：
  - `UNSW-NB15`
  - 更长训练轮次
  - 小规模超参数优化

### 5.5 服务器打包回传

打包：

```powershell
.\scripts\Package-ServerResults.ps1
```

回传：

- ToDesk 文件传输
- 或：
  - [Fetch-ServerResults.ps1](/d:/Study/研2/spaper_zj/scripts/Fetch-ServerResults.ps1)

---

## 6. 标准实验顺序

这是当前推荐的正式实验顺序，不是历史顺序。

### 阶段 A：环境验证

1. 检查 `torch + cuda`
2. 跑 `BiAT-MLP` smoke test
3. 跑 `BiAT-FTTransformer` smoke test

### 阶段 B：主结果

#### 数据集一：UNSW-NB15

模型：

- `XGBoost`
- `LightGBM`
- `LSTM`
- `BiAT-MLP`
- `BiAT-FTTransformer`

目标：

- 验证新主线是否比旧 RL 线明显更强
- 看 `FT-Transformer` 是否比 `MLP` 更稳

#### 数据集二：CIC-IDS2017-random

模型：

- `HGBT`
- `XGBoost`
- `LightGBM`
- `LSTM`
- `BiAT-MLP`
- `BiAT-FTTransformer`

目标：

- 验证新主线在第二个现代基准上的成立程度
- 观察与树模型的差距是否缩小

### 阶段 C：补充实验

#### NSL-KDD

用途：

- 保留旧 RL 线结果
- 生成消融
- 生成鲁棒性分析

#### CICIoT2023 / grouped

用途：

- 展示项目已经接入更新数据集
- 作为补充现代数据集
- 当前不作为主证明集

---

## 7. 结果汇总与绘图工作流

### 7.1 结果文件

所有正式结果统一落到：

```text
outputs/results/
```

主要包括：

- `main_results_summary_*.csv`
- `main_results_detailed_*.csv`
- `ablation_results*.csv`
- `adversarial_robustness*.csv`
- 单模型 `*.json`

### 7.2 统一汇总脚本

主结果：

- [evaluate_main_results.py](/d:/Study/研2/spaper_zj/scripts/evaluate_main_results.py)

消融：

- [evaluate_ablation.py](/d:/Study/研2/spaper_zj/scripts/evaluate_ablation.py)

鲁棒性：

- [evaluate_adversarial_robustness.py](/d:/Study/研2/spaper_zj/scripts/evaluate_adversarial_robustness.py)

### 7.3 绘图

统一绘图入口：

- [plot_all_figures.py](/d:/Study/研2/spaper_zj/scripts/plot_all_figures.py)

当前输出：

- `outputs/figures/`
- `latex_source/figures/generated/`

绘图原则：

- 主图以均值趋势为主
- 不确定性保留在表格
- 导出 `PDF + PNG`
- 版式适配双栏论文

---

## 8. 论文写作工作流

### 8.1 论文源文件

主入口：

- [main.tex](/d:/Study/研2/spaper_zj/latex_source/main.tex)

关键章节：

- [01_introduction.tex](/d:/Study/研2/spaper_zj/latex_source/sections/01_introduction.tex)
- [02_related_work.tex](/d:/Study/研2/spaper_zj/latex_source/sections/02_related_work.tex)
- [03_methodology.tex](/d:/Study/研2/spaper_zj/latex_source/sections/03_methodology.tex)
- [04_experiments.tex](/d:/Study/研2/spaper_zj/latex_source/sections/04_experiments.tex)
- [05_conclusion.tex](/d:/Study/研2/spaper_zj/latex_source/sections/05_conclusion.tex)

### 8.2 当前写作原则

1. 不再把旧 RL 线写成现代数据集上的强主结论
2. `NSL-KDD` 作为受控验证
3. `UNSW-NB15 + CIC-IDS2017` 作为现代主实验
4. `CICIoT2023` 作为补充
5. 结论必须与真实结果一致，不夸大鲁棒性和 FPR 优势

### 8.3 文档维护分工

#### 工作流文档

- 本文档负责：
  - 完整步骤
  - 当前主线
  - 推荐执行顺序
  - 项目级流程

#### 优化日志文档

- [optimization_decision_log.md](/d:/Study/研2/spaper_zj/docs/optimization_decision_log.md) 负责：
  - 问题
  - 失败
  - 修复
  - 判断过程
  - 路线转向原因

---

## 9. 当前推荐执行清单

如果现在重新接手项目，推荐按这个顺序推进：

1. 看本文档，建立全局流程认知
2. 看 `optimization_decision_log.md`，理解路线为什么会变化
3. 在本地检查最新代码和文档
4. 在服务器上：
   - `git pull`
   - 准备数据
   - 跑 `Suite core`
5. 回传 `outputs/results / outputs/models / outputs/figures`
6. 在本地：
   - 汇总结果
   - 重画图
   - 更新实验章节
7. 最后整理投稿版本

---

## 9.1 当前最优先待办

如果只看最近一轮，优先级如下：

1. 等待服务器 `Suite core` 跑完
2. 回传并检查：
   - `outputs/results`
   - `outputs/models`
   - `outputs/figures`
3. 重算：
   - `UNSW-NB15`
   - `CIC-IDS2017-random`
   的主结果表
4. 以 `BiAT-FTTransformer` 为当前主方法，更新论文口径
5. 使用双卡并行脚本做 `UNSW-NB15 / BiAT-FTTransformer` 更长 epoch 重训

---

## 10. 当前最重要的现实结论

### 已经成立的

- 旧 `Bi-ARL` 只在 `NSL-KDD` 上能支撑相对弱的正面结论
- 新路线 `BiAT` 明显比旧 RL 主线更合理
- `UNSW-NB15` 是当前最关键的现代主实验集
- Windows 服务器批量实验链路已经打通
- `BiAT-FTTransformer` 已经成为当前最好的神经网络主方法

### 还没完全成立的

- 新主线还没有全面超过强树模型
- `CIC-IDS2017-random` 上 `BiAT-FTTransformer` 只比 `BiAT-MLP` 小幅提升
- `CICIoT2023` 当前更适合作为补充，而不是主证明集

---

## 11. 相关关键文件

### 主代码

- [config.py](/d:/Study/研2/spaper_zj/src/utils/config.py)
- [data_loader.py](/d:/Study/研2/spaper_zj/src/utils/data_loader.py)
- [bilevel_supervised_ids.py](/d:/Study/研2/spaper_zj/src/baselines/bilevel_supervised_ids.py)
- [bilevel_fttransformer_ids.py](/d:/Study/研2/spaper_zj/src/baselines/bilevel_fttransformer_ids.py)
- [evaluate_main_results.py](/d:/Study/研2/spaper_zj/scripts/evaluate_main_results.py)

### 服务器脚本

- [Run-ServerSuite.ps1](/d:/Study/研2/spaper_zj/scripts/Run-ServerSuite.ps1)
- [Package-ServerResults.ps1](/d:/Study/研2/spaper_zj/scripts/Package-ServerResults.ps1)
- [server_conda_setup.md](/d:/Study/研2/spaper_zj/docs/server_conda_setup.md)

### 记录文档

- [optimization_decision_log.md](/d:/Study/研2/spaper_zj/docs/optimization_decision_log.md)
- [experiment_writing_log_2026-03-08.md](/d:/Study/研2/spaper_zj/docs/experiment_writing_log_2026-03-08.md)
- [project_structure_overview.md](/d:/Study/研2/spaper_zj/docs/project_structure_overview.md)

---

**文档版本**: v2.0  
**最后更新**: 2026-03-09  
**当前状态**: 以路线 C（BiAT）为主线，旧 Bi-ARL 作为历史与补充结果保留
