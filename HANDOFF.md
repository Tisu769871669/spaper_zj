# 给下一窗口的交接说明

## 一、任务总结

### 项目背景
- **项目**：小论文，中英文 LaTeX 双版，目标 **CCF-A 安全会议/期刊**（如 IEEE S&P、CCS、USENIX Security、NDSS）。
- **方法**：**BiAT 框架**（Bilevel Adversarial Training）：双层对抗训练 + RL（Bi-ARL）与梯度式（BiAT-MLP、BiAT-FTTransformer）。全文已统一用「BiAT 框架」，不再使用「Route-C」。
- **环境**：Conda 环境 **`spaper`**，有 CUDA；模型与结果在 **`outputs/`** 下。

### 已完成的实验补强（按 Phase）
| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 在 `evaluate_main_results.py` 中注册 Transformer-IDS 模型 | ✅ 完成 |
| 1 | 对 nsl-kdd、unsw-nb15、cic-ids2017-random 跑主结果评估，生成 per-seed 详细 CSV（含 ±std） | ✅ 完成 |
| 2 | 对 nsl-kdd、unsw-nb15 跑 FGSM + PGD 对抗鲁棒性评估 | ✅ 完成 |
| 3 | 在 UNSW-NB15 上训练 Vanilla FT-Transformer（4 种子），与 BiAT-FTT 对比 | ✅ 完成 |
| 3b | 若差距不足则调参重训 BiAT-FTT（条件触发） | ❌ 已取消 |
| 4 | 运行统计显著性检验（Wilcoxon + Cohen's d） | ✅ 完成 |
| 5 | 超参敏感性：K_inner、λ | ✅ 完成 |
| 6 | 用真实实验数字回填中英文论文表格并重新编译 | ✅ 完成 |

### 其他已做事项
- 引用从约 15 条扩到 **47 条**（数据集原文、FT-Transformer、双层优化、PGD/MARL、IDS 等）；不在论文里写未做实验或编造 ±std。
- 全文去掉「Route-C」改为 BiAT；清除虚假 ±std 和虚假基线，只保留真实跑出的数字。
- 在相关脚本（如 `evaluate_adversarial_robustness.py` 等）中增加进度显示，避免误以为卡死。

---

## 二、结果总结

### 主结果
- 主结果表已用**真实** ±std 回填；已加入 XGBoost、LightGBM 对比。
- 主结果评估脚本：`scripts/evaluate_main_results.py`；结果 CSV 在 `outputs/results/main_results_detailed_*.csv` 等。

### 对抗鲁棒性
- FGSM 与 PGD 结果文件均已生成：`robustness_summary_nsl_kdd.csv`、`robustness_summary_unsw_nb15.csv`。
- 正文主表保留 FGSM 汇总，PGD 结果以正文讨论 + CSV 补充形式呈现。
- **注意**：NSL-KDD 上的 BiAT-MLP 鲁棒性仅为单种子冒烟测试，正文与表题中需显式标注，不能误写成 4 种子均值。

### Vanilla FT-Transformer 基线
- UNSW-NB15 上 4 种子已训；结论约为「约 1% F1 换约 16.3pp 鲁棒性提升」。

### 统计显著性
- 已报 Wilcoxon p 与 Cohen's d（如 BiAT-FTT vs BiAT-MLP 大效应量）；脚本：`scripts/run_significance_tests.py`。

### 超参敏感性
- **K_inner**：已有 K_inner=1,3,5,10 的结果表并写入论文。
- **λ 敏感性**：已完成，结果文件为 `outputs/results/sensitivity_lambda_unsw_nb15.csv`。论文应写成定量结论：$\lambda$ 从 0.3 增大到 0.9 时，F1 下降、FPR 上升。

### 论文与编译
- 英文：`latex_source/main.tex`，节在 `latex_source/sections/01_introduction.tex` … `05_conclusion.tex`，参考文献 `latex_source/references.bib`。
- 中文：`latex_source_cn/main_cn.tex`。
- 两版 PDF 已能编译通过（英文约 8 页，中文约 12 页）。

---

## 三、关键路径速查

| 用途 | 路径 |
|------|------|
| 英文论文节 | `latex_source/sections/04_experiments.tex`（实验表） |
| 中文论文 | `latex_source_cn/main_cn.tex` |
| 主结果评估 | `scripts/evaluate_main_results.py` |
| 对抗鲁棒性 | `scripts/evaluate_adversarial_robustness.py` |
| 显著性检验 | `scripts/run_significance_tests.py` |
| 超参敏感性 | `scripts/run_hyperparam_sensitivity.py`（`--experiment k_inner` / `lambda`） |
| 结果 CSV | `outputs/results/` |
| 模型查找 | `src/utils/config.py`，`Config.find_model_file()` |

---

## 四、建议下一窗口优先核对/补做

1. **表格一致性**：确认 ±std、Transformer-IDS 行、鲁棒性（FGSM/PGD）、显著性、K_inner/λ 均来自真实实验且与 CSV 一致，不保留“future work”式过时措辞。
2. **单种子说明**：NSL-KDD 上 BiAT-MLP 鲁棒性是单种子 smoke test，表题与正文都要显式说明。
3. **现代性补强**：若继续扩实验，优先补更现代数据集或 baseline，而不是重复旧 RL 结果。
4. 所有命令在 **`conda activate spaper`** 下执行。

---

## 五、Antigravity 审查与修复记录（2026-03-12）

### 审查范围
对整个项目进行了全面审查：英文论文全五章 + `references.bib`（47条引用）、中文论文全文（494行）、核心代码模块（`bilevel_trainer.py`、`fgsm.py`、`bilevel_fttransformer_ids.py`、`bilevel_supervised_ids.py`、`config.py`）、全部 CSV 实验结果。已完成数据表格与论文的交叉验证。

### 发现的问题与已执行的修复

#### P0 严重问题（3 项，已全部修复）

| # | 问题 | 修改文件 | 变更内容 |
|---|------|---------|---------|
| 1 | 英文 Abstract 遗留内部术语 `route-C` | `latex_source/main.tex` | 删除 `route-C models`，改为 `yielding BiAT-MLP and BiAT-FTTransformer` |
| 2 | 英文结论把已完成实验仍标为 future steps | `latex_source/sections/05_conclusion.tex` | 重写最后一段：已完成的工作改为总结，仅保留真正的未来方向 |
| 3 | 中文论文数据集列表包含 CSE-CIC-IDS2018 但无实验结果 | `latex_source_cn/main_cn.tex` | 从数据集列表中移除该条 |

#### P1 中等问题（4 项，已全部修复）

| # | 问题 | 修改文件 | 变更内容 |
|---|------|---------|---------|
| 4 | 中文 UNSW-NB15 鲁棒性表缺少 MARL/Vanilla PPO 行 | `latex_source_cn/main_cn.tex` | 补充两行使中英文一致 |
| 5 | 英文 UNSW 表 BiAT-FTTransformer F1 被错误加粗 | `latex_source/sections/04_experiments.tex` | 取消 F1 列加粗，仅保留 FPR 列 |
| 6 | 代码注释仍写 `route-C refactor` | `src/baselines/bilevel_supervised_ids.py` | 更新为 `BiAT framework` |
| 7 | 中英文 label 仍用 `routec` 命名 | `03_methodology.tex` + `main_cn.tex` | 统一更名为 `biat` |

#### P2 小问题（1 项已修复 + 2 项建议手动）

| # | 问题 | 修改文件 | 变更内容 |
|---|------|---------|---------|
| 8 | `ben2009robust` BibTeX 类型错误 | `latex_source/references.bib` | `@article` 改为 `@book` |
| 9 | `src/dataset/` 空目录 | — | 建议删除 |
| 10 | `src/read_docx_temp.py` 临时文件 | — | 建议删除 |

### 数据一致性验证
全部 CSV 实验数据与论文表格数字经交叉验证**一致**（考虑合理四舍五入）。

### 代码质量
- `bilevel_trainer.py`：双层训练逻辑正确
- `fgsm.py`：FGSM/PGD 攻击实现符合标准
- `bilevel_fttransformer_ids.py`：FT-Transformer 架构正确
- `config.py`：多数据集配置管理合理
