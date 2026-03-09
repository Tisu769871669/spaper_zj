# 训练输出整理说明

## 📁 问题

训练完成后,模型文件保存在 `src/` 目录下,而不是预期的 `outputs/` 目录。

## ✅ 解决方案

使用整理脚本自动将文件移动到正确位置:

### 方法1: 运行整理脚本(推荐)

```powershell
# 训练完成后运行
cd D:\Study\研2\小论文
conda activate spaper
python scripts/organize_outputs.py
```

这会自动将所有文件移动到 `outputs/` 目录的正确位置。

---

### 方法2: 手动整理

如果需要手动整理,请按以下结构移动文件:

**从 `src/` 移动到 `outputs/models/`:**

```
src/attacker_bilevel_seed42.pth  →  outputs/models/BiARL/seed42/attacker.pth
src/defender_bilevel_seed42.pth  →  outputs/models/BiARL/seed42/defender.pth
src/defender_vanilla_ppo_seed42.pth  →  outputs/models/VanillaPPO/seed42/model.pth
src/lstm_ids_seed42.pth  →  outputs/models/LSTM/seed42/model.pth

(同样处理 seed 101, 202)
```

**从 `src/checkpoints/` 移动到 `outputs/checkpoints/`:**

```
src/checkpoints/seed42/  →  outputs/checkpoints/BiARL/seed42/
```

**实验结果:**

```
src/experiment_results.csv  →  outputs/results/experiment_results.csv
```

---

## 📊 整理后的目录结构

```
outputs/
├── models/
│   ├── BiARL/
│   │   ├── seed42/
│   │   │   ├── attacker.pth
│   │   │   └── defender.pth
│   │   ├── seed101/
│   │   └── seed202/
│   ├── VanillaPPO/
│   │   ├── seed42/model.pth
│   │   ├── seed101/model.pth
│   │   └── seed202/model.pth
│   └── LSTM/
│       ├── seed42/model.pth
│       ├── seed101/model.pth
│       └── seed202/model.pth
├── checkpoints/
│   └── BiARL/
│       ├── seed42/
│       ├── seed101/
│       └── seed202/
└── results/
    └── experiment_results.csv
```

---

## 🔄 更新 START_TRAINING.md

训练完成后,记得添加这一步:

```powershell
# 1. 训练完成
python scripts/train_all_models.py

# 2. 整理输出文件
python scripts/organize_outputs.py

# 3. 运行评估
python src/experiments.py --seed 42
```

---

## ⚠️ 注意

- 整理脚本会**移动**(不是复制)文件
- 建议在训练完成并确认无误后再运行
- 脚本会自动创建必要的目录结构
