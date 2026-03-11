# 当前状态与下一步

这个文档只用于一件事：

- 以后重开窗口时，最快速地恢复当前项目状态

如果时间有限，优先看这个文件；如果需要细节，再看：

- [BI_ARL_WORKFLOW.md](/d:/Study/研2/spaper_zj/BI_ARL_WORKFLOW.md)
- [optimization_decision_log.md](/d:/Study/研2/spaper_zj/docs/optimization_decision_log.md)

---

## 1. 当前项目主线

项目已经从最初的 `Bi-ARL`（纯 RL 检测器）演化成两条线：

- 旧主线：`Bi-ARL`
  - 主要在 `NSL-KDD` 上成立
- 新主线：路线 C
  - `BiAT-MLP`
  - `BiAT-FTTransformer`

当前真正的主方法是：

- `BiAT-FTTransformer`

---

## 2. 当前最重要结果

### NSL-KDD

- `Bi-ARL`
  - `F1 = 80.17%`
  - `FPR = 10.41%`

结论：

- 旧 RL 主线在受控 benchmark 上成立

### UNSW-NB15

- `BiAT-FTTransformer`
  - `F1 = 89.51%`
  - `FPR = 14.97%`

结论：

- 这是当前最强神经网络主结果
- 并且在该数据集上取得了所有对照方法中最低的 `FPR`
- 但总体 `F1` 仍低于 `LightGBM`

### CIC-IDS2017-random

- `BiAT-FTTransformer`
  - `F1 = 87.09%`
  - `FPR = 6.78%`

结论：

- 最优 `UNSW` 配置迁移后仍有小幅正收益
- 但整体仍明显落后于树模型

---

## 3. 当前论文口径

现在论文不能写成：

- SOTA
- 全面优于现代 baseline
- 已解决现代 IDS 鲁棒性问题

现在最稳的口径是：

- `Bi-ARL` 在 `NSL-KDD` 上对普通 RL 有效
- 路线 C 说明同样的 bi-level 对抗训练思想迁移到更强 detector 后，现代数据结果明显改善
- `BiAT-FTTransformer` 是当前最强神经网络方案
- 在 `UNSW-NB15` 上拿到了 `89.51% F1 / 14.97% FPR`
- 但整体仍落后于最强 boosted-tree baseline

---

## 4. 已经完成的关键文档

- 工作流总文档：
  - [BI_ARL_WORKFLOW.md](/d:/Study/研2/spaper_zj/BI_ARL_WORKFLOW.md)
- 决策日志：
  - [optimization_decision_log.md](/d:/Study/研2/spaper_zj/docs/optimization_decision_log.md)
- 导师总结：
  - [advisor_summary_cn.md](/d:/Study/研2/spaper_zj/docs/advisor_summary_cn.md)
- 中文汇报稿：
  - [oral_report_cn.md](/d:/Study/研2/spaper_zj/docs/oral_report_cn.md)
- 中文摘要与创新点：
  - [abstract_contributions_cn.md](/d:/Study/研2/spaper_zj/docs/abstract_contributions_cn.md)
- 答辩/PPT 提纲：
  - [defense_ppt_outline_cn.md](/d:/Study/研2/spaper_zj/docs/defense_ppt_outline_cn.md)
- 当前论文 PDF：
  - [main.pdf](/d:/Study/研2/spaper_zj/latex_source/main.pdf)

---

## 5. 如果重开窗口，先做什么

按这个顺序：

1. 看 [NEXT_STEPS.md](/d:/Study/研2/spaper_zj/docs/NEXT_STEPS.md)
2. 看 [BI_ARL_WORKFLOW.md](/d:/Study/研2/spaper_zj/BI_ARL_WORKFLOW.md)
3. 看 [optimization_decision_log.md](/d:/Study/研2/spaper_zj/docs/optimization_decision_log.md)
4. 看最新论文 [main.pdf](/d:/Study/研2/spaper_zj/latex_source/main.pdf)

---

## 6. 下一步优先级

如果继续推进，推荐顺序：

1. 不再盲目扩实验，优先做投稿/答辩整理
2. 如果还要补实验，优先补更现代的数据集或更现代 baseline
3. 不优先继续重跑旧 `Bi-ARL`
4. 不优先继续在 `CICIoT2023` 上刷分

---

## 7. 当前最值得继续补的方向

### 方向 A：投稿与答辩材料

- 5 分钟汇报稿
- 10 分钟汇报稿
- 逐页 PPT 文案

### 方向 B：现代性补强

- 更现代数据集
  - `CSE-CIC-IDS2018`
  - `NF-UQ-NIDS / SHIELD`
  - `5G-NIDD`
- 更现代 baseline
  - SSL
  - GNN
  - 更新的 transformer / calibration baseline

---

## 8. 不要重复做的事

- 不要再把 `NSL-KDD` 当主结论数据集
- 不要再把旧 `Bi-ARL` 包装成现代最优方法
- 不要再直接运行有 bug 的旧版 `sweep` 脚本
- 不要把 `data/` 直接传到 git 仓库

---

## 9. 仓库状态

- 当前主仓库：
  - `D:\Study\研2\spaper_zj`
- 远程仓库：
  - `https://github.com/Tisu769871669/spaper_zj.git`

如果服务器继续跑实验，先：

```powershell
git pull origin main
```

---

## 10. 一句话总结

当前项目已经从一个半成品 RL 原型，推进成了一篇围绕“双层对抗训练框架”的完整论文草稿。  
最强结果是 `UNSW-NB15` 上的 `BiAT-FTTransformer: 89.51% F1 / 14.97% FPR`。  
下一步更适合做投稿与答辩收口，而不是继续盲目堆实验。
