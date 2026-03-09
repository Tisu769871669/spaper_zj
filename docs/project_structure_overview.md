# 项目目录树与文件说明

本文档用于说明当前项目的目录结构、各目录用途，以及关键文件的功能。  
说明原则：

- 优先介绍对论文、实验、训练、评测真正有用的文件
- 对自动生成文件（如 `.pyc`、`events.out.tfevents`、LaTeX 中间文件）按类别说明
- 对结果文件按“用途”分类说明，而不逐个重复解释格式相同的文件

---

## 1. 项目根目录

```text
小论文 - 副本/
├─ .agent/
├─ .vscode/
├─ data/
├─ docs/
├─ latex_source/
├─ outputs/
├─ runs/
├─ scripts/
├─ src/
├─ BI_ARL_WORKFLOW.md
├─ README.md
├─ requirements.txt
├─ START_TRAINING.md
└─ SYSTEM_PROMPT.md
```

### 根目录文件说明

- `BI_ARL_WORKFLOW.md`
  - 项目的原始工作流说明，偏开发流程文档。

- `README.md`
  - 项目的总说明文档，介绍项目目标、结构、运行方式和当前状态。

- `requirements.txt`
  - Python 依赖列表。当前已包含 `torch / sklearn / pyarrow / xgboost / lightgbm` 等实验依赖。

- `START_TRAINING.md`
  - 面向训练启动的快速说明文档。

- `SYSTEM_PROMPT.md`
  - 与开发辅助流程有关的提示词文档，不属于论文方法本体。

### 根目录文件夹说明

- `.agent/`
  - 辅助代理相关目录，包含本地 skill 配置，不属于论文核心内容。

- `.vscode/`
  - VS Code 工作区配置目录。

- `data/`
  - 数据集目录，包含 `NSL-KDD / UNSW-NB15 / CIC-IDS2017` 数据文件。

- `docs/`
  - 项目文档目录，包含实验记录、投稿状态、数据集说明、训练输出说明等。

- `latex_source/`
  - 论文 LaTeX 源码目录。

- `outputs/`
  - 训练模型、检查点、日志、结果表等统一输出目录。

- `runs/`
  - TensorBoard 运行日志目录。

- `scripts/`
  - 实验驱动脚本、评测脚本、下载脚本。

- `src/`
  - 主要源码目录，包含 agent、环境、算法、基线、工具模块。

---

## 2. data/

```text
data/
├─ KDDTrain+.txt
├─ KDDTest+.txt
├─ UNSW_NB15_training-set.csv
├─ UNSW_NB15_testing-set.csv
└─ CIC_IDS2017_machine_learning/
   ├─ Monday-WorkingHours....parquet
   ├─ Tuesday-WorkingHours....parquet
   ├─ Wednesday-workingHours....parquet
   ├─ Thursday-WorkingHours-Morning-WebAttacks....parquet
   ├─ Thursday-WorkingHours-Afternoon-Infilteration....parquet
   ├─ Friday-WorkingHours-Morning....parquet
   ├─ Friday-WorkingHours-Afternoon-PortScan....parquet
   └─ Friday-WorkingHours-Afternoon-DDos....parquet
```

### 文件说明

- `KDDTrain+.txt`
  - `NSL-KDD` 训练集。

- `KDDTest+.txt`
  - `NSL-KDD` 测试集。

- `UNSW_NB15_training-set.csv`
  - `UNSW-NB15` 官方训练集。

- `UNSW_NB15_testing-set.csv`
  - `UNSW-NB15` 官方测试集。

- `CIC_IDS2017_machine_learning/*.parquet`
  - `CIC-IDS2017` 的公开 parquet 镜像版本，供表格建模直接使用。
  - 当前项目支持两种协议：
    - 按工作日划分：`cic-ids2017`
    - 随机分层划分：`cic-ids2017-random`

---

## 3. docs/

```text
docs/
├─ cic_ids2017_setup.md
├─ config_usage.md
├─ experiment_writing_log_2026-03-08.md
├─ proposal_extracted.txt
├─ round3_submission_notes.md
├─ submission_status_2026-03-08.md
├─ training_outputs.md
└─ unsw_nb15_setup.md
```

### 文件说明

- `cic_ids2017_setup.md`
  - `CIC-IDS2017` 的下载、划分方式、运行示例说明。

- `config_usage.md`
  - 配置文件使用说明，帮助理解 `Config` 中的路径、数据集、输出方式。

- `experiment_writing_log_2026-03-08.md`
  - 当前最详细的中文工作记录。
  - 记录了修复内容、训练命令、实验结果、结论和后续建议。

- `proposal_extracted.txt`
  - 从其他材料中抽取出的文本，通常用于参考提案或早期写作素材。

- `round3_submission_notes.md`
  - 第三轮整理出的投稿方向、相关论文、数据集和基线建议文档。

- `submission_status_2026-03-08.md`
  - 当前投稿状态简表。
  - 适合快速看：这篇论文现在能投到什么程度，还差什么。

- `training_outputs.md`
  - 训练输出说明文档，介绍 `outputs/` 里的内容组织方式。

- `unsw_nb15_setup.md`
  - `UNSW-NB15` 的下载、放置、训练和评测说明。

---

## 4. latex_source/

```text
latex_source/
├─ figures/
│  ├─ comparison.png
│  ├─ generated/
│  │  ├─ ablation/
│  │  ├─ main_results/
│  │  └─ robustness/
│  ├─ robustness.png
│  └─ robustness_stress.png
├─ out/
├─ sections/
│  ├─ 01_introduction.tex
│  ├─ 02_related_work.tex
│  ├─ 03_methodology.tex
│  ├─ 04_experiments.tex
│  └─ 05_conclusion.tex
├─ main.tex
├─ references.bib
├─ main.pdf
└─ main.aux / main.bbl / main.blg / main.fls / main.log / ...
```

### 核心文件说明

- `main.tex`
  - 论文主入口文件。
  - 负责文档类、摘要、关键词、章节引入、参考文献等。

- `references.bib`
  - BibTeX 参考文献数据库。

- `main.pdf`
  - 当前编译出的论文 PDF。

### sections/

- `01_introduction.tex`
  - 引言，包含问题背景、动机、贡献点。

- `02_related_work.tex`
  - 相关工作，覆盖传统 IDS、现代深度模型、泛化、鲁棒性和对抗 RL。

- `03_methodology.tex`
  - 方法章节，说明 Bi-ARL 的问题定义、奖励设计、PPO 与 bi-level 训练逻辑。

- `04_experiments.tex`
  - 实验章节，当前已经覆盖：
    - `NSL-KDD`
    - `UNSW-NB15`
    - `CIC-IDS2017-random`
    - `CIC-IDS2017` 严格日期划分

- `05_conclusion.tex`
  - 结论章节，当前口径已经统一为“方法有效但局限明显”的诚实版本。

### figures/

- `comparison.png`
  - 模型对比图。

- `generated/`
  - 由 Python 绘图脚本自动生成的论文图目录。
  - 当前会同步保存 `PDF + PNG` 两种格式，便于论文插图与快速预览。

- `robustness.png`
  - 鲁棒性相关图。

- `robustness_stress.png`
  - 压力测试/鲁棒性扩展图。

### 其他文件

- `main.aux / main.bbl / main.blg / main.fls / main.log / main.fdb_latexmk`
  - LaTeX 编译中间文件。
  - 这些不是手工维护的核心内容。

- `out/`
  - LaTeX 编译输出目录。

---

## 5. scripts/

```text
scripts/
├─ batch_evaluate_ablation.py
├─ download_cic_ids2017.py
├─ evaluate_ablation.py
├─ evaluate_ablation_simple.py
├─ evaluate_adversarial_robustness.py
├─ evaluate_fpr_optimization.py
├─ evaluate_main_results.py
├─ generate_analysis.py
├─ plot_ablation.py
├─ plot_all_figures.py
├─ plot_main_results.py
├─ plot_robustness.py
├─ plotting_utils.py
├─ organize_outputs.py
├─ train_all_models.py
├─ ORGANIZE_OUTPUTS.md
└─ README.md
```

### 文件说明

- `download_cic_ids2017.py`
  - 下载 `CIC-IDS2017` parquet 镜像的脚本。

- `evaluate_main_results.py`
  - 主结果评测脚本。
  - 当前支持：
    - RL 模型
    - `LSTM-IDS`
    - `HGBT-IDS`
    - `XGBoost-IDS`
    - `LightGBM-IDS`

- `evaluate_ablation.py`
  - 消融实验评测脚本。

- `evaluate_ablation_simple.py`
  - 更简化版本的消融评测脚本，通常用于快速检查。

- `evaluate_adversarial_robustness.py`
  - 对抗鲁棒性评测脚本，主要用于 `NSL-KDD` 的 FGSM 结果。

- `evaluate_fpr_optimization.py`
  - 与 `FPR` 优化相关的额外分析脚本。

- `batch_evaluate_ablation.py`
  - 批量运行消融相关评估。

- `generate_analysis.py`
  - 用于生成分析性结果或辅助输出。

- `plot_main_results.py`
  - 主结果绘图脚本。
  - 从 `outputs/results/main_results_summary*.csv` 自动生成各数据集主结果图，以及跨数据集对比图。

- `plot_ablation.py`
  - 消融实验绘图脚本。
  - 当前会绘制 `NSL-KDD` 和 `UNSW-NB15` 的 clean/stress 对比图。

- `plot_robustness.py`
  - 鲁棒性绘图脚本。
  - 当前会根据 `adversarial_robustness.csv` 生成 `FGSM epsilon - performance` 曲线图。

- `plot_all_figures.py`
  - 总入口脚本。
  - 运行一次即可批量生成主结果图、消融图和鲁棒性图。

- `plotting_utils.py`
  - 绘图公共模块。
  - 统一管理配色、输出目录、保存格式和排序逻辑，保证图风格一致。

- `organize_outputs.py`
  - 整理 `outputs/` 目录结构的脚本。

- `train_all_models.py`
  - 统一训练入口脚本，可按数据集批量训练各模型。

- `ORGANIZE_OUTPUTS.md`
  - `outputs/` 整理说明文档。

- `README.md`
  - `scripts/` 子目录下的说明文档。

---

## 6. src/

```text
src/
├─ agents/
├─ algorithms/
├─ attacks/
├─ baselines/
├─ checkpoints/
├─ dataset/
├─ envs/
├─ utils/
├─ experiments.py
├─ experiment_results.csv
├─ main_train.py
├─ main_train_bilevel.py
├─ read_docx_temp.py
├─ run_paper_experiments.py
├─ structure.md
└─ __init__.py
```

### 顶层源码文件

- `main_train.py`
  - 标准多智能体/非 bi-level 训练入口。

- `main_train_bilevel.py`
  - `Bi-ARL` 主训练入口，是论文方法对应的核心训练脚本。

- `experiments.py`
  - 早期实验驱动脚本，保留作为辅助入口。

- `run_paper_experiments.py`
  - 论文实验统一运行脚本，偏旧但仍可参考。

- `experiment_results.csv`
  - 早期结果文件，历史遗留。

- `read_docx_temp.py`
  - 临时脚本，不属于核心训练或评测主线。

- `structure.md`
  - `src/` 目录结构说明，早期文档。

### agents/

```text
src/agents/
├─ attacker_agent.py
├─ defender_agent.py
└─ __init__.py
```

- `attacker_agent.py`
  - 攻击者策略网络定义。

- `defender_agent.py`
  - 防御者策略网络定义。
  - 当前已经补齐 `forward()`，可直接用于评测与推理。

### algorithms/

```text
src/algorithms/
├─ bilevel_trainer.py
└─ __init__.py
```

- `bilevel_trainer.py`
  - Bi-level 训练器核心实现。
  - 负责 inner loop / outer loop 的嵌套训练逻辑。

### attacks/

```text
src/attacks/
├─ fgsm.py
└─ __init__.py
```

- `fgsm.py`
  - FGSM 攻击实现，用于鲁棒性评测。

### baselines/

```text
src/baselines/
├─ bilevel_fixed_attacker.py
├─ hgbt_ids.py
├─ lightgbm_ids.py
├─ lstm_ids.py
├─ marl_baseline.py
├─ vanilla_ppo.py
├─ xgboost_ids.py
├─ README.md
└─ __init__.py
```

- `vanilla_ppo.py`
  - 不带对抗训练的单智能体 PPO 基线。

- `marl_baseline.py`
  - 同时更新 attacker/defender 的 MARL 基线。

- `bilevel_fixed_attacker.py`
  - 固定攻击者版本，用于消融 `w/o Inner Loop`。

- `lstm_ids.py`
  - 监督学习 LSTM 基线。

- `hgbt_ids.py`
  - Histogram Gradient Boosting 表格基线。

- `xgboost_ids.py`
  - XGBoost 表格基线。

- `lightgbm_ids.py`
  - LightGBM 表格基线。

- `README.md`
  - baseline 子目录说明。

### envs/

```text
src/envs/
├─ network_security_game.py
└─ __init__.py
```

- `network_security_game.py`
  - 攻防环境核心文件。
  - 负责：
    - 数据样本采样
    - 攻击者扰动
    - 防御者动作映射
    - 奖励计算
    - 状态转移

### utils/

```text
src/utils/
├─ config.py
├─ data_loader.py
├─ output_manager.py
├─ ppo.py
├─ result_analyzer.py
├─ statistical_tests.py
├─ training_logger.py
└─ __init__.py
```

- `config.py`
  - 全局配置中心。
  - 当前支持：
    - `nsl-kdd`
    - `unsw-nb15`
    - `cic-ids2017`
    - `cic-ids2017-random`

- `data_loader.py`
  - 多数据集 loader 核心文件。
  - 当前支持：
    - `NSL-KDD`
    - `UNSW-NB15`
    - `CIC-IDS2017` 日期划分
    - `CIC-IDS2017` 随机分层划分

- `ppo.py`
  - PPO 实现，是 RL 训练的基础优化器。

- `output_manager.py`
  - 输出文件路径和结果组织辅助。

- `result_analyzer.py`
  - 结果分析工具。

- `statistical_tests.py`
  - 统计检验工具。

- `training_logger.py`
  - 训练日志记录辅助。

### 其他目录

- `checkpoints/`
  - 旧版源码目录下保留的检查点目录。

- `dataset/`
  - 目前基本不是主路径，更多是历史结构遗留。

- `__pycache__/`
  - Python 编译缓存，无需手工维护。

---

## 7. outputs/

```text
outputs/
├─ checkpoints/
├─ figures/
├─ logs/
├─ models/
├─ results/
└─ README.md
```

### 总体说明

`outputs/` 是当前实验结果的主存放目录，是项目里最重要的“运行产物”目录。

### checkpoints/

- 存放训练中间检查点。
- 按模型和数据集组织。

### logs/

- 存放训练日志。
- 一般与模型训练过程对应。

### figures/

- 存放论文级实验图。
- 当前由 `scripts/plot_all_figures.py` 统一生成。
- 目录按用途拆分为：
  - `main_results/`
  - `ablation/`
  - `robustness/`
- 每张图同时导出：
  - `PDF`：用于论文投稿和矢量插图
  - `PNG`：用于快速预览、汇报和文档嵌入

### models/

- 存放最终模型权重。
- 目前已经按数据集隔离，例如：
  - `nsl_kdd`
  - `unsw_nb15`
  - `cic_ids2017`
  - `cic_ids2017_random`

### results/

这是最关键的结果目录，主要包括以下几类：

- 主结果表
  - `main_results_summary*.csv`
  - `main_results_detailed*.csv`

- 消融结果
  - `ablation_results*.csv`
  - `ablation_results_detailed*.csv`

- 鲁棒性结果
  - `adversarial_robustness*.csv`

- 单模型 baseline 结果
  - `hgbt_baseline_*.json`
  - `xgboost_baseline_*.json`
  - `lightgbm_baseline_*.json`

### 当前最关键的结果文件

- `main_results_summary.csv`
  - `NSL-KDD` 主结果

- `main_results_summary_unsw_nb15.csv`
  - `UNSW-NB15` 主结果

- `main_results_summary_cic_ids2017_random.csv`
  - `CIC-IDS2017` 随机分层主结果

- `main_results_summary_cic_ids2017.csv`
  - `CIC-IDS2017` 日期划分更严格实验结果

---

## 8. runs/

```text
runs/
└─ bilevel_seed*/
   └─ events.out.tfevents...
```

### 文件说明

- 这是 TensorBoard 日志目录。
- 主要用于查看训练曲线和过程监控。
- 不属于论文主文档，但对分析训练稳定性有帮助。

---

## 9. 哪些文件最值得优先看

如果你是从“论文投稿”和“项目接手”角度看，这几个文件优先级最高：

### 第一组：论文与状态

- `latex_source/main.tex`
- `latex_source/sections/04_experiments.tex`
- `docs/experiment_writing_log_2026-03-08.md`
- `docs/submission_status_2026-03-08.md`

### 第二组：训练与评测主线

- `src/main_train_bilevel.py`
- `src/algorithms/bilevel_trainer.py`
- `src/envs/network_security_game.py`
- `scripts/evaluate_main_results.py`
- `scripts/evaluate_ablation.py`

### 第三组：数据与 baseline

- `src/utils/config.py`
- `src/utils/data_loader.py`
- `src/baselines/lstm_ids.py`
- `src/baselines/hgbt_ids.py`
- `src/baselines/xgboost_ids.py`
- `src/baselines/lightgbm_ids.py`

### 第四组：最终结果

- `outputs/results/main_results_summary.csv`
- `outputs/results/main_results_summary_unsw_nb15.csv`
- `outputs/results/main_results_summary_cic_ids2017_random.csv`
- `outputs/results/main_results_summary_cic_ids2017.csv`

---

## 10. 当前项目一句话总结

这个项目现在已经形成了一条比较完整的链路：

- 数据加载
- 攻防环境
- Bi-level RL 训练
- 多类 baseline
- 多数据集评测
- 论文 LaTeX 写作
- 结果记录与投稿状态文档

它已经不是一个简单的代码仓库，而是一个“带论文产出的实验型研究项目”。
