# 🚀 训练启动指南

## 📋 准备工作

### 1. 确认环境

```powershell
# 进入项目目录
cd D:\Study\研2\小论文

# 激活conda环境
conda activate spaper

# 验证CUDA可用
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

应该看到: `CUDA可用: True` ✅

---

## 🎯 开始训练

### 一键启动完整训练管道

```powershell
cd D:\Study\研2\小论文
conda activate spaper
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/train_all_models.py
```

### 或者简化版(一行命令)

```powershell
cd D:\Study\研2\小论文; conda activate spaper; $env:KMP_DUPLICATE_LIB_OK="TRUE"; python scripts/train_all_models.py
```

---

## 📊 训练内容

将依次训练以下模型(每个模型3个seeds):

1. **Bi-ARL** (100 episodes × 3 seeds)
2. **Vanilla PPO** (100 episodes × 3 seeds)  
3. **LSTM-IDS** (20 epochs × 3 seeds)

**总计**: 9个训练任务

---

## ⏱️ 预计时间

| 模型 | 单次训练 | 3个seeds |
|------|---------|----------|
| Bi-ARL | ~8-10分钟 | ~25-30分钟 |
| Vanilla PPO | ~6-8分钟 | ~18-24分钟 |
| LSTM-IDS | ~2-3分钟 | ~6-9分钟 |

**总计**: 约 **50-65分钟** (GPU加速)

---

## 📁 输出文件位置

训练完成后,所有模型将保存在:

```
src/
├── attacker_bilevel_seed42.pth
├── defender_bilevel_seed42.pth
├── defender_vanilla_ppo_seed42.pth
├── lstm_ids_seed42.pth
├── (同样的 seed 101, 202...)
└── checkpoints/
    └── seed42/
        ├── checkpoint_ep50.pth
        └── checkpoint_ep100.pth
```

---

## 🔍 监控训练进度

训练会实时输出:

- ✅ 每个episode的进度
- ✅ 内层/外层循环的奖励
- ✅ 模型保存确认
- ✅ TensorBoard日志(如已安装)

**查看TensorBoard** (可选):

```powershell
# 新开一个终端
cd D:\Study\研2\小论文
conda activate spaper
tensorboard --logdir=runs/bilevel
```

然后访问: <http://localhost:6006>

---

## ✅ 训练完成后

### 1. 验证模型文件

```powershell
# 检查是否生成了所有模型
ls src/*.pth
```

应该看到18个.pth文件(3 seeds × 6 models)

### 2. 运行评估

```powershell
python src/experiments.py --seed 42
```

这会生成:

- `src/experiment_results.csv` - 所有模型的性能对比
- 统计显著性分析

### 3. 查看结果

```powershell
# 查看CSV结果
cat src/experiment_results.csv

# 或用Excel打开
start src/experiment_results.csv
```

---

## ⚠️ 常见问题

### Q: 训练中断了怎么办?

A: 检查 `src/checkpoints/` 目录,可以从最近的检查点继续

### Q: CUDA不可用?

A: 确保在 `spaper` 环境下运行,该环境已配置CUDA支持

### Q: 想只训练某个模型?

A: 可以单独运行:

```powershell
# 只训练Bi-ARL
python src/main_train_bilevel.py --seed 42 --episodes 100

# 只训练Vanilla PPO
python src/baselines/vanilla_ppo.py --seed 42 --episodes 100

# 只训练LSTM
python src/baselines/lstm_ids.py --seed 42 --epochs 20
```

---

## 📞 需要帮助?

查看详细日志:

```powershell
# 训练日志会实时显示在终端
# 如果需要保存日志:
python scripts/train_all_models.py > training_log.txt 2>&1
```

---

**准备好了吗?**

复制下面的命令开始训练:

```powershell
cd D:\Study\研2\小论文; conda activate spaper; $env:KMP_DUPLICATE_LIB_OK="TRUE"; python scripts/train_all_models.py
```

🚀 **祝训练顺利!**
