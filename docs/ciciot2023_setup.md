# CICIoT2023 数据集接入说明

## 1. 数据集概览

- 数据集名称：`CICIoT2023`
- 时间：`2023`
- 官方页面：<https://www.unb.ca/cic/datasets/iotdataset-2023.html>
- 推荐引用：
  - Neto et al., *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment*, 2023

## 2. 这篇工作做了什么

- 在包含 `105` 个设备的 IoT 拓扑中采集流量
- 构造了 `33` 种攻击
- 提供适合实时 IoT 入侵检测研究的大规模数据集和基准

## 3. 当前项目中的使用方式

当前项目将 `CICIoT2023` 作为：

- 面向 `CCF-A` 重构路线的“更新数据集”
- 强监督基线（`XGBoost / LightGBM / HGBT`）的扩展评测对象
- 后续 `bi-level adversarial training for supervised IDS` 的关键验证集

## 4. 目录放置方式

将官方解压后的 CSV 文件放到：

```text
data/
└─ CICIoT2023/
   ├─ BenignTraffic*.csv
   ├─ DDoS-*.csv
   ├─ DoS-*.csv
   └─ ...
```

当前 loader 会递归扫描 `data/CICIoT2023/` 下的所有 CSV 文件。

## 5. 标签约定

`CICIoT2023` 的原始 CSV 通常按攻击类别分文件保存，因此当前项目使用文件名推断二分类标签：

- 文件名包含 `benign` -> 标签 `0`
- 其他攻击文件 -> 标签 `1`

这适合当前论文的二分类 IDS 设定。

## 6. 划分方式

当前实现使用：

- 随机分层划分
- 默认 `train/test = 80/20`
- 为了控制资源开销，默认抽样上限：
  - train: `240000`
  - test: `120000`

## 7. 冒烟测试样例

如果你暂时不想下载完整官方数据，可以先运行：

```bash
python scripts/download_ciciot2023_sample.py
```

它会下载一个非常小的公开镜像样例到 `data/CICIoT2023/`，用于验证 loader 和 baseline 管线。

注意：

- 这个样例不是完整官方数据
- 不能用于论文主结果
- 只能用于代码联通性检查

## 8. 运行示例

```bash
python src/baselines/xgboost_ids.py --dataset ciciot2023 --seed 42
python src/baselines/lightgbm_ids.py --dataset ciciot2023 --seed 42
python scripts/evaluate_main_results.py --dataset ciciot2023
```
