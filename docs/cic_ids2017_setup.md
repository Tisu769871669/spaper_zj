# CIC-IDS2017 接入说明

## 数据来源

当前项目使用的是一个公开镜像版本：

- Hugging Face: `bvsam/cic-ids-2017`

该镜像将官方 `MachineLearningCVE` CSV 转换为 parquet，更适合直接做表格建模。

## 下载命令

在项目根目录执行：

```bash
python scripts/download_cic_ids2017.py
```

下载后的文件目录：

- `data/CIC_IDS2017_machine_learning/`

## 当前项目中的划分方式

项目里目前支持两种划分方式。

### 1. `cic-ids2017`

这是更严格的按工作日划分：

- 训练集：
  - `Monday-WorkingHours`
  - `Tuesday-WorkingHours`
  - `Wednesday-workingHours`
  - `Thursday-WorkingHours-Morning-WebAttacks`
  - `Thursday-WorkingHours-Afternoon-Infilteration`

- 测试集：
  - `Friday-WorkingHours-Morning`
  - `Friday-WorkingHours-Afternoon-PortScan`
  - `Friday-WorkingHours-Afternoon-DDos`

这属于更难的日期迁移设定，因此结果通常会明显弱于随机划分。

### 2. `cic-ids2017-random`

这是更常规的随机分层划分：

- 将 8 个 parquet 文件先合并
- 再按标签做随机分层切分
- 默认测试集比例 `20%`

这更适合生成论文主表，因为它与很多现有表格模型论文的常规设置更接近。

## 计算量控制

为了让当前项目先跑通，`CIC-IDS2017` loader 默认采用可复现下采样：

- 训练样本上限：`300000`
- 测试样本上限：`150000`

对应配置在：

- `src/utils/config.py`

如果后续租服务器或者本机内存更充足，可以提高这个上限。

## 已接入的基线

- `HGBT-IDS`
- `XGBoost-IDS`
- `LightGBM-IDS`

运行示例：

```bash
python src/baselines/hgbt_ids.py --dataset cic-ids2017 --seed 42
python src/baselines/xgboost_ids.py --dataset cic-ids2017 --seed 42
python src/baselines/lightgbm_ids.py --dataset cic-ids2017 --seed 42

python src/baselines/hgbt_ids.py --dataset cic-ids2017-random --seed 42
python src/baselines/xgboost_ids.py --dataset cic-ids2017-random --seed 42
python src/baselines/lightgbm_ids.py --dataset cic-ids2017-random --seed 42
```

如果要汇总主结果：

```bash
python scripts/evaluate_main_results.py --dataset cic-ids2017
python scripts/evaluate_main_results.py --dataset cic-ids2017-random
```

注意：

- 该脚本会自动训练 `HGBT / XGBoost / LightGBM`
- RL 和 LSTM 如果尚未训练，会被自动跳过
