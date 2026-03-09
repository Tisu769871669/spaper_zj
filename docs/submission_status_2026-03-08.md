# 投稿状态简表（2026-03-08）

## 当前结论

这篇论文目前已经从“半成品工程”推进到了“可继续打磨的研究草稿”，但还不属于强投稿版本。

最准确的定位是：

- 一个以 `Bi-level RL` 为核心思想的网络入侵检测研究原型
- 在 `NSL-KDD` 上有正面结果
- 在 `UNSW-NB15` 和 `CIC-IDS2017` 上暴露出现代数据集稳定性与竞争力不足的问题

## 目前最能成立的贡献

1. 提出并实现了一个 attacker-defender 的 bi-level RL 训练框架
2. 在 `NSL-KDD` 上，`Bi-ARL` 相对 `Vanilla PPO` 有明确改进
3. 通过 `UNSW-NB15` 和 `CIC-IDS2017` 证明当前方法并未在现代数据集上稳定成立
4. 论文现在的优点是“诚实、完整、有实验支撑”，而不是“数值全面领先”

## 目前最薄弱的地方

1. 现代数据集上，RL 结果不够强
2. 强 baseline 越补越说明树模型很强
3. 目前还没有 Transformer / SSL / GNN 一类近期深度基线
4. `CIC-IDS2017-random` 上树模型近乎饱和，使 RL 很难作为主角

## 现在适合的论文叙述

建议采用下面这种口径：

- 本文研究的是 `bi-level adversarial RL` 在 IDS 中的可行性
- 该方法在受控 benchmark 上优于普通 RL
- 但在更现代、更严格的数据集设定下仍明显落后于强监督表格模型
- 因此本文的价值主要在于方法框架与负面结果分析，而不是 SOTA 性能

## 现在不适合的论文叙述

不建议再写：

- state of the art
- robust IDS solution
- lowest FPR overall
- superior across datasets
- strong adversarial robustness

## 如果现在就投稿

更适合：

- 偏方法探索、实验诚实的小论文
- 允许 negative / nuanced result 的场景

不适合：

- 高强度 benchmark 导向的顶级投稿
- 需要强现代 SOTA 对比的 venue

## 如果继续提升一轮

最值得做的 3 件事：

1. 再补一个更现代的 baseline
   - `Transformer` 优先
   - 或者至少补一个公开实现的 `SSL/GNN`

2. 再补一个现代数据集或更真实协议
   - `CSE-CIC-IDS2018`
   - 或真实的跨日/跨场景测试

3. 明确论文故事
   - 如果保 `Bi-ARL` 主线，就突出“controlled RL gain + modern limitation”
   - 不要再尝试包装成全面最优模型

## 当前建议

如果你的目标是“尽快形成一版能交出去的稿子”，下一步最应该做的是：

- 继续精修论文语言和表格
- 准备中文汇报版和导师讨论版
- 只补最少量、最有价值的实验

如果你的目标是“冲更强投稿”，那就必须继续补现代深度 baseline 和更强的跨分布实验。
