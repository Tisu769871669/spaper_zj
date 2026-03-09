# 优化决策过程记录

本文档专门记录项目推进过程中每次遇到关键问题时，我们是如何判断、如何修改、为什么这么修改的。

目的不是记录所有命令，而是保留：

- 问题是什么
- 为什么原方案不够
- 我们的优化思路是什么
- 修改后带来了什么结果
- 下一步为什么这样走

---

## 1. 旧 RL 主线只在 NSL-KDD 上成立

### 问题

最初的 `Bi-ARL (PPO defender)` 只在 `NSL-KDD` 上相对优于 `Vanilla PPO`，但在：

- `UNSW-NB15`
- `CIC-IDS2017`

上没有形成正面主结果。

### 判断

这说明问题不在论文表述，而在方法本身：

- 检测器 backbone 太弱
- RL 在现代 tabular IDS 上并不是最自然的检测器
- 审稿人会直接质疑泛化性和现实意义

### 优化思路

保留：

- bi-level 思想
- attacker-defender 对抗训练
- worst-case robustness 动机

放弃：

- “检测器必须是 PPO policy” 这件事

改为：

- `bi-level adversarial training for supervised IDS`

### 结果

项目主线从 “RL-based detector” 切换为：

- “强监督检测器 + 双层对抗训练”

这是路线 C 的起点。

---

## 2. 先用 MLP 做第一版可运行原型

### 问题

在切换主线之后，不能一上来就写复杂模型，否则会同时引入：

- 新训练框架
- 新 backbone
- 新评测逻辑

这样很难定位问题。

### 判断

需要先有一个最小可运行版本，验证：

- 双层对抗训练是否真的能在现代数据集上跑通
- 相比旧 RL 主线是否至少有明显提升

### 优化思路

先实现：

- `BiAT-MLP`

结构：

- inner loop：对输入做多步扰动
- outer loop：MLP 在 clean + adversarial loss 上训练

### 结果

在 `UNSW-NB15` 上，`BiAT-MLP` 的 4 seeds 结果为：

- Recall = `93.34%`
- Precision = `80.24%`
- F1 = `86.29%`
- FPR = `28.19%`

这说明：

- 新主线显著强于旧 RL 主线
- 而且方差明显更可控

---

## 3. CICIoT2023 接入后出现“几乎满分”

### 问题

在完整 `CICIoT2023` 上，`XGBoost / LightGBM / BiAT-MLP` 都接近：

- `F1 = 1.0`
- `FPR = 0.0`

### 判断

这不代表方法已经完美，而更可能意味着：

- 当前划分协议过于宽松
- train/test 存在明显同源分布
- 二分类任务本身太容易

如果不处理，论文会被质疑“协议太水”。

### 优化思路

新增更严格的协议：

- `ciciot2023-grouped`

核心原则：

- 同一源文件的流量不同时进入 train/test
- 尽量减少同源样本泄露

### 结果

即使在 `grouped` 协议下，结果依然接近满分。

### 结论

说明问题不只是随机切分，而是：

- 当前这版 `CICIoT2023` 二分类任务本身就偏容易

因此它可以作为：

- 现代补充数据集

但不适合单独承担论文主结论。

---

## 4. 下载大数据集时频繁超时

### 问题

`CICIoT2023` 全量镜像下载时出现：

- 长时间读取超时
- 中断后重复下载
- 部分文件坏分片
- `Range` 请求不稳定

### 判断

这不是数据集不可用，而是下载策略太脆弱。

### 优化思路

把下载脚本升级为：

- 可切换镜像基址
- 断点续传
- 自动重试
- 发现坏分片时自动删除并重下

### 结果

完整 `CICIoT2023` 已下载到：

- `data/CICIoT2023/`

总量：

- `166` 个 CSV
- 约 `6.576 GB`

---

## 5. 结果图排版重叠

### 问题

论文中的：

- `ablation`
- `cross-dataset`
- `robustness`

图在缩小预览时出现标题、图例、主图区域重叠。

### 判断

不是内容错，而是版式不适合双栏论文。

### 优化思路

- 图例去重
- 图例和标题分层
- 留出固定顶部空白
- 图尺寸收紧
- 统一论文风格

### 结果

结果图已统一到：

- `outputs/figures/`
- `latex_source/figures/generated/`

并且已经接入论文。

---

## 6. 主结果图里的误差棒太扎眼

### 问题

主结果柱状图里的误差棒在小尺寸预览下非常突兀，影响阅读。

### 判断

虽然 `CCF-A` 稿件常会展示不确定性，但：

- 现在表格里已经有 `mean ± std`
- 图的主要任务是表达趋势

### 优化思路

- 主结果图去掉误差棒
- 标准差留在表格中

### 结果

图更接近正式投稿版的常见做法：

- 图负责趋势
- 表负责精确统计

---

## 7. 当前阶段的总判断

### 成功的优化

- 从旧 RL 主线切换到路线 C
- 完成 `BiAT-MLP`
- 在 `UNSW-NB15` 上得到稳定、可写的正向结果
- 补齐 `CICIoT2023` 与更严格协议验证

### 仍未解决的问题

- `BiAT-MLP` 在 `CIC-IDS2017-random` 上明显落后于树模型
- 说明 backbone 还不够强

### 因此下一步

继续升级到：

- `FT-Transformer + bi-level adversarial training`

这一步的目标不是“换模型试试”，而是系统验证：

- 更强 tabular backbone 是否能缩小与 `XGBoost / LightGBM` 的差距

---

## 8. GPU 不工作，定位到环境问题

### 问题

虽然机器本身有：

- `RTX 4060`

而且 `nvidia-smi` 可以正常识别显卡，但项目训练一直跑在 CPU 上。

### 判断

检查后发现问题不在代码，而在 Python 环境：

- 当前默认环境中的 `torch` 是 CPU 版
- `torch.cuda.is_available() = False`
- `torch.version.cuda = None`

### 优化思路

不在默认环境里强行重装一套新 `torch`，而是直接使用已有的 GPU 环境：

- conda 环境：`spaper`

检查结果：

- Python: `3.10.19`
- Torch: `2.5.1+cu121`
- `torch.cuda.is_available() = True`

### 执行策略

以后所有基于 `torch` 的训练与评测，统一优先使用：

- `D:\python\Anaconda3\envs\spaper\python.exe`

而不是默认的 base 环境。

### 结果

`BiAT-FTTransformer` 已经成功在 GPU 上跑通，`Config.DEVICE` 显示为：

- `cuda:0`

这意味着后续：

- `BiAT-MLP`
- `BiAT-FTTransformer`
- `LSTM`
- 其他 `torch` 模型

都可以切换到 GPU 环境执行。

需要说明：

- `XGBoost / LightGBM / HGBT` 这些树模型仍主要走 CPU，不属于环境异常。

---

## 9. 本地 4060 负责开发，服务器 GPU 负责整套实验

### 问题

虽然本地已经可以使用 GPU，但长时间批量实验仍有两个问题：

- `BiAT-FTTransformer` 需要逐种子串行跑，多数据集时总时长很长
- 本地机器还承担开发、写作、绘图，不适合持续被实验占满

### 判断

最合理的分工是：

- 本地 `4060`
  - 开发
  - 冒烟测试
  - 快速超参数验证
- 组里服务器 GPU（当前已确认可用 `Tesla V100-PCIE-32GB`）
  - 多种子正式实验
  - 更长训练轮次
  - 批量评测
  - 最终结果打包回传

### 优化思路

把服务器流程标准化，避免每次手工敲命令：

- 同步代码和数据到服务器
- 服务器端批量运行
- 服务器端统一打包结果
- 本地统一拉回结果归档

### 已实施

新增脚本：

- `scripts/sync_project_to_server.sh`
- `scripts/run_server_suite.sh`
- `scripts/package_server_results.sh`
- `scripts/fetch_server_results.sh`
- `scripts/Sync-ProjectToServer.ps1`
- `scripts/Run-ServerSuite.ps1`
- `scripts/Package-ServerResults.ps1`
- `scripts/Fetch-ServerResults.ps1`

并更新文档：

- `docs/server_conda_setup.md`
- `docs/project_structure_overview.md`

### 结果

实验链路现在可以明确分成两层：

- 本地负责改代码和小规模验证
- 服务器负责整套论文实验

这样更符合后续冲 `CCF-A` 的节奏，也便于最后统一汇总。
