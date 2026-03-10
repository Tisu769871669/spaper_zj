# 服务器 Conda 环境与实验运行说明

本文档用于把项目部署到组里的 Windows GPU 服务器（如 `V100 / A100`）并运行实验。

适用场景：

- 服务器使用 Windows + `conda`
- 需要在服务器上重新创建环境
- 需要跑 `BiAT-MLP / BiAT-FTTransformer / LSTM / 树模型`

---

## 1. 克隆项目

```powershell
git clone https://github.com/Tisu769871669/spaper_zj.git
cd spaper_zj
```

如果你准备从本地把代码和 `data/` 直接同步到服务器，也可以用：

```powershell
.\scripts\Sync-ProjectToServer.ps1 -ServerUser your_name -ServerHost your.server.edu.cn -RemoteDir ~/spaper_zj
```

默认行为：

- 同步代码
- 同步 `data/`
- 跳过 `.git / outputs / runs / latex_source/out`

如果只同步代码，不同步数据：

```powershell
.\scripts\Sync-ProjectToServer.ps1 -ServerUser your_name -ServerHost your.server.edu.cn -RemoteDir ~/spaper_zj -SkipData
```

---

## 2. 创建 conda 环境

建议环境名直接用：

- `spaper`

```powershell
conda create -n spaper python=3.10 -y
conda activate spaper
```

说明：

- 当前本地验证通过的 GPU 环境也是 `python 3.10`
- 不建议在服务器上直接用 `python 3.12`

---

## 3. 安装 PyTorch GPU 版

优先安装与你服务器 CUDA 驱动兼容的官方版本。

如果服务器驱动兼容 `CUDA 12.1`，可直接用：

```powershell
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装后检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出应包含：

- `True`
- 你的 GPU 名称（例如 `Tesla V100-PCIE-32GB`）

---

## 4. 安装项目依赖

```powershell
pip install -r requirements.txt
```

如果个别包没装全，至少补上：

```powershell
pip install pyarrow xgboost lightgbm tqdm seaborn tensorboard
```

说明：

- `pyarrow`：读取 `CIC-IDS2017` 的 parquet 必需
- `xgboost / lightgbm`：强基线必需

---

## 5. 数据准备

本仓库默认 **不上传 `data/`**，所以服务器上需要你自行准备数据。

### 5.1 NSL-KDD

放到：

```text
data/
├─ KDDTrain+.txt
└─ KDDTest+.txt
```

### 5.2 UNSW-NB15

放到：

```text
data/
├─ UNSW_NB15_training-set.csv
└─ UNSW_NB15_testing-set.csv
```

### 5.3 CIC-IDS2017

如果你要跑 `CIC-IDS2017`，可以直接用项目脚本下载镜像：

```powershell
python scripts/download_cic_ids2017.py
```

下载后应存在：

```text
data/CIC_IDS2017_machine_learning/
```

### 5.4 CICIoT2023

如果你要跑 `CICIoT2023`，用：

```powershell
python scripts/download_ciciot2023_full.py
```

下载后应存在：

```text
data/CICIoT2023/
```

---

## 6. 推荐先跑哪些实验

如果你先想验证环境没问题，建议按这个顺序：

### 6.1 树模型 baseline

```powershell
python src/baselines/xgboost_ids.py --dataset unsw-nb15 --seed 42
python src/baselines/lightgbm_ids.py --dataset unsw-nb15 --seed 42
```

### 6.2 新主线：BiAT-MLP

```powershell
python src/baselines/bilevel_supervised_ids.py --dataset unsw-nb15 --seed 42 --epochs 5 --batch_size 512 --epsilon 0.02 --alpha 0.005 --steps 2 --adv_weight 0.6
```

### 6.3 升级 backbone：BiAT-FTTransformer

```powershell
python src/baselines/bilevel_fttransformer_ids.py --dataset unsw-nb15 --seed 42 --epochs 5 --batch_size 512 --epsilon 0.02 --alpha 0.005 --steps 2 --adv_weight 0.6
```

---

## 7. 多种子运行建议

服务器上建议按种子串行跑，不要在单卡上同时起多个 `FT-Transformer` 训练。

推荐种子：

- `42`
- `3407`
- `8888`
- `123`

例如：

```powershell
python src/baselines/bilevel_fttransformer_ids.py --dataset unsw-nb15 --seed 42 --epochs 5 --batch_size 512 --epsilon 0.02 --alpha 0.005 --steps 2 --adv_weight 0.6
python src/baselines/bilevel_fttransformer_ids.py --dataset unsw-nb15 --seed 3407 --epochs 5 --batch_size 512 --epsilon 0.02 --alpha 0.005 --steps 2 --adv_weight 0.6
python src/baselines/bilevel_fttransformer_ids.py --dataset unsw-nb15 --seed 8888 --epochs 5 --batch_size 512 --epsilon 0.02 --alpha 0.005 --steps 2 --adv_weight 0.6
python src/baselines/bilevel_fttransformer_ids.py --dataset unsw-nb15 --seed 123 --epochs 5 --batch_size 512 --epsilon 0.02 --alpha 0.005 --steps 2 --adv_weight 0.6
```

---

## 8. 批量运行整套实验

推荐直接使用服务器批量脚本，而不是手工一条条输入。

### 8.1 跑核心实验

```powershell
conda activate spaper
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite core
```

默认包含：

- `UNSW-NB15`
  - `HGBT / XGBoost / LightGBM / LSTM`
  - `BiAT-MLP / BiAT-FTTransformer`
- `CIC-IDS2017-random`
  - `HGBT / XGBoost / LightGBM / LSTM`
  - `BiAT-MLP`
  - `BiAT-FTTransformer`
- 汇总主结果
- 自动绘图

### 8.2 跑完整套实验

```powershell
conda activate spaper
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite all
```

完整套实验在 `core` 基础上还会加：

- `CICIoT2023-grouped` 补充实验
- `NSL-KDD` 的主结果、消融和鲁棒性评测

### 8.3 常用环境变量

```powershell
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite core
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite all -DisableFtCic
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite all -Seeds 42,3407
```

例如只跑核心实验，但跳过 `CIC-IDS2017-random` 上的 FT-Transformer：

```powershell
.\scripts\Run-ServerSuite.ps1 -CondaEnv spaper -Suite core -DisableFtCic
```

服务器运行日志默认保存在：

```text
outputs/logs/server_runs/
```

---

## 8.4 双卡并行优化实验

如果服务器上有两张 GPU，并且你要继续优化 `UNSW-NB15` 上的路线 C，推荐使用专门的双卡脚本，而不是重新跑整套 `core`。

### 8.4.1 单配置强化训练

当前推荐先用这一条做正式强化训练：

```powershell
conda activate spaper
.\scripts\Run-DualGpuOptimization.ps1 -CondaEnv spaper -Model ft -Dataset unsw-nb15 -Epochs 15 -BatchSize 1024 -Epsilon 0.02 -Alpha 0.005 -Steps 2 -AdvWeight 0.6 -LearningRate 0.0007 -WeightDecay 0.0001 -Dropout 0.15 -DToken 64 -Seeds 42,3407,8888,123 -GpuIds 0,1
```

说明：

- 这是当前最稳妥的第二轮优化起点
- 相比旧配置，优先提高 `batch_size`
- 目标是更充分利用 `V100 32GB`

### 8.4.2 三配置 sweep

如果服务器时间充足，并且你希望直接做一轮小范围搜索，可以用：

```powershell
conda activate spaper
.\scripts\Run-FtOptimizationSweep.ps1 -CondaEnv spaper -Dataset unsw-nb15 -Seeds 42,3407,8888,123 -GpuIds 0,1
```

这轮 sweep 默认会依次跑三组配置：

- `ft_unsw_bs1024_lr7e4_aw060`
- `ft_unsw_bs1024_lr7e4_aw065`
- `ft_unsw_bs1024_lr5e4_eps015`

每组结束后，会自动保存带配置名的结果文件：

- `outputs/results/main_results_summary_unsw_nb15_<config>.csv`
- `outputs/results/main_results_detailed_unsw_nb15_<config>.csv`

因此不会覆盖前一组的对比结果。

脚本：

- [Run-DualGpuOptimization.ps1](/d:/Study/研2/spaper_zj/scripts/Run-DualGpuOptimization.ps1)

推荐命令：

```powershell
conda activate spaper
.\scripts\Run-DualGpuOptimization.ps1 -CondaEnv spaper -Model ft -Dataset unsw-nb15 -Epochs 15 -Seeds 42,3407,8888,123 -GpuIds 0,1
```

说明：

- `-Model ft`
  - 只优化 `BiAT-FTTransformer`
- `-Model mlp`
  - 只优化 `BiAT-MLP`
- `-Model both`
  - 两个路线 C 模型都跑

这个脚本的策略不是单模型多卡，而是：

- `GPU 0` 跑一部分种子
- `GPU 1` 跑另一部分种子

这比对当前 tabular 模型使用 `DataParallel` 更实用。

---

## 9. 结果汇总

实验结果主要保存在：

```text
outputs/results/
outputs/models/
```

统一汇总脚本：

```powershell
python scripts/evaluate_main_results.py --dataset unsw-nb15
python scripts/evaluate_main_results.py --dataset cic-ids2017-random
```

---

## 10. 结果打包与拿回本地

服务器上实验结束后，推荐先打包：

```powershell
conda activate spaper
.\scripts\Package-ServerResults.ps1
```

默认会生成：

```text
outputs\packages\spaper_results_<hostname>_<timestamp>.zip
```

然后在本地拉回：

```powershell
.\scripts\Fetch-ServerResults.ps1 -ServerUser your_name -ServerHost your.server.edu.cn -RemoteDir ~/spaper_zj
```

默认下载到：

```text
outputs/server_fetch/
```

建议只同步这些目录：

```text
outputs/results/
outputs/models/
```

如果用 `scp`，例如：

```powershell
scp -r user@server:/path/to/spaper_zj/outputs/results .\outputs\
scp -r user@server:/path/to/spaper_zj/outputs/models .\outputs\
```

---

## 11. 当前推荐主实验

如果你在服务器上优先跑最有价值的内容，建议按这个顺序：

1. `UNSW-NB15`
   - `BiAT-MLP`
   - `BiAT-FTTransformer`
   - `XGBoost`
   - `LightGBM`

2. `CIC-IDS2017-random`
   - `BiAT-MLP`
   - `BiAT-FTTransformer`
   - `XGBoost`
   - `LightGBM`

3. `CICIoT2023`
   - 只作为补充现代数据集
   - 不建议把随机划分结果作为主结论

---

## 12. 当前最重要的原则

- `torch` 模型统一用 GPU 环境
- 树模型主要跑 CPU，不需要强行占 GPU
- `FT-Transformer` 不要在单卡上并行跑多个种子
- 先保证 `UNSW-NB15` 结果稳定，再扩其他数据集
