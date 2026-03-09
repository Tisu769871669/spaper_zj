# 训练输出文件说明

本项目的训练过程会生成以下文件:

## 📁 文件结构

```
项目根目录/
├── src/
│   ├── *.pth                          # 训练好的模型文件
│   ├── experiment_results.csv         # 评估结果汇总
│   └── checkpoints/                   # 训练检查点(可选)
│       └── seed42/
│           ├── checkpoint_ep50.pth
│           └── checkpoint_ep100.pth
│
└── logs/                              # JSON训练日志(新增)
    ├── BiARL/
    │   ├── bilevel_seed42.json
    │   ├── bilevel_seed101.json
    │   └── bilevel_seed202.json
    ├── VanillaPPO/
    │   └── vanilla_ppo_seed42.json
    ├── LSTM/
    │   └── lstm_ids_seed42.json
    └── training_summary.json          # 所有实验汇总
```

---

## 📄 文件详细说明

### 1. 模型文件 (*.pth)

**位置**: `src/*.pth`

**命名规则**:
- `{model_name}_seed{seed}.pth`
- 例如: `defender_bilevel_seed42.pth`

**内容**: PyTorch模型的state_dict,包含所有神经网络参数

**用途**:
- 加载模型进行评估
- 继续训练
- 部署使用

**大小**: 约60-70KB

---

### 2. 实验结果CSV (experiment_results.csv)

**位置**: `src/experiment_results.csv`

**格式**:
```csv
Seed,Model,Condition,Acc,Recall,Precision,F1,FPR
42,Bi-ARL,Clean,0.7518,0.7110,0.7642,0.7642,0.1943
42,Bi-ARL,Adversarial,0.7460,0.7110,0.7580,0.7580,0.2030
42,RandomForest,Clean,0.7704,0.6183,0.7541,0.7541,0.0285
```

**用途**:
- 生成论文表格
- 统计显著性检验
- 可视化对比

---

### 3. 训练日志JSON (新增)

**位置**: `logs/{ModelName}/{experiment_name}.json`

**格式**:
```json
{
  "experiment_name": "bilevel_seed42",
  "start_time": "2026-01-14T23:15:00",
  "config": {
    "seed": 42,
    "lr": 0.0003,
    "episodes": 100,
    "device": "cuda:0"
  },
  "episodes": [
    {
      "episode": 1,
      "timestamp": "2026-01-14T23:15:05",
      "inner_avg_reward": 10.5,
      "outer_reward": -5.0,
      "inner_steps": 5,
      "inner_converged": false
    },
    ...
  ],
  "summary": {
    "total_time_seconds": 600,
    "convergence_rate": 0.45,
    "final_model_path": "src/defender_bilevel_seed42.pth"
  },
  "end_time": "2026-01-14T23:25:00"
}
```

**用途**:
- 详细分析训练过程
- 绘制训练曲线
- 调试和优化
- 可复现性验证

---

### 4. 检查点文件 (checkpoint_ep*.pth)

**位置**: `src/checkpoints/seed{seed}/checkpoint_ep{episode}.pth`

**内容**:
```python
{
    'episode': 50,
    'attacker_state_dict': {...},
    'defender_state_dict': {...},
    'ppo_attacker_optimizer': {...},
    'ppo_defender_optimizer': {...},
    'inner_loop_converged_count': 20
}
```

**用途**:
- 从中断处恢复训练
- 分析训练中间状态
- 调试问题

---

## 🔧 使用JSON日志

### 查看训练日志
```python
import json

with open('logs/BiARL/bilevel_seed42.json', 'r') as f:
    log = json.load(f)

print(f"训练时长: {log['summary']['total_time_seconds']}秒")
print(f"总episode数: {len(log['episodes'])}")

# 查看第10个episode
print(log['episodes'][9])
```

### 绘制训练曲线
```python
import matplotlib.pyplot as plt

episodes = [ep['episode'] for ep in log['episodes']]
rewards = [ep['outer_reward'] for ep in log['episodes']]

plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
```

---

## 📊 文件大小估算

| 文件类型 | 大小 | 数量 |
|---------|------|------|
| 模型文件(.pth) | ~65KB | 18个(3 seeds × 6 models) |
| 实验结果CSV | ~10KB | 1个 |
| JSON日志 | ~50KB | 9个(3 seeds × 3 models) |
| 检查点 | ~130KB | 每个seed 2个 |

**总计**: 约1.5MB
