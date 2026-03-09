# Config使用示例和说明

## 基本用法

### 导入配置

```python
from src.utils.config import Config

# 使用配置
device = Config.DEVICE
lr = Config.LR
seeds = Config.SEEDS
```

### 设置随机种子

```python
Config.set_seed(42)  # 设置全局随机种子
```

### 使用路径配置

```python
# 获取数据路径
train_data = Config.TRAIN_DATA
test_data = Config.TEST_DATA

# 获取模型保存路径
model_path = Config.get_model_path("BiARL", seed=42, model_name="defender")
# 返回: outputs/models/BiARL/seed42/defender.pth

# 获取检查点路径
checkpoint = Config.get_checkpoint_path("BiARL", seed=42, episode=50)
# 返回: outputs/checkpoints/BiARL/seed42/checkpoint_ep50.pth

# 获取结果CSV
results_csv = Config.get_results_csv()
# 返回: outputs/results/experiment_results.csv
```

### 智能模型查找(向后兼容)

```python
# 自动查找模型(支持新旧路径)
model_path = Config.find_model_file("BiARL", seed=42, model_name="defender")
# 会按优先级查找:
# 1. outputs/models/BiARL/seed42/defender.pth
# 2. src/defender_bilevel_seed42.pth
# 3. src/defender.pth
```

## 配置项说明

### 实验配置

- `SEEDS = [42, 101, 202]` - 所有实验使用的种子列表
- `DEFAULT_SEED = 42` - 默认种子
- `RL_EPISODES = 100` - RL训练轮数
- `LSTM_EPOCHS = 20` - LSTM训练周期

### 路径配置

所有输出统一管理在`outputs/`下:

- `MODELS_DIR` - 模型保存目录
- `LOGS_DIR` - 训练日志目录
- `RESULTS_DIR` - 实验结果目录
- `CHECKPOINTS_DIR` - 检查点目录
- `FIGURES_DIR` - 图表目录

### 训练参数

- `LR = 3e-4` - 学习率
- `GAMMA = 0.99` - 折扣因子
- `EPS_CLIP = 0.2` - PPO裁剪参数
- `K_EPOCHS = 4` - PPO更新epoch数

### Bi-level参数

- `INNER_LOOP_STEPS = 5` - 内层循环最大步数
- `KL_THRESHOLD = 0.01` - KL散度阈值
- `USE_BILEVEL = True` - 是否使用双层优化

## 修改脚本使用Config

### 示例1: 训练脚本

```python
from src.utils.config import Config

# 使用配置中的seeds
for seed in Config.SEEDS:
    Config.set_seed(seed)
    
    # 使用配置中的训练参数
    train(episodes=Config.RL_EPISODES)
    
    # 使用配置中的路径
    save_path = Config.get_model_path("BiARL", seed, "defender")  
    torch.save(model.state_dict(), save_path)
```

### 示例2: 评估脚本

```python
from src.utils.config import Config

# 使用配置中的seeds
for seed in Config.SEEDS:
    # 加载模型(自动查找)
    model_path = Config.find_model_file("BiARL", seed, "defender")
    model.load_state_dict(torch.load(model_path))
    
    # 评估并保存结果
    results = evaluate(model)
    results_csv = Config.get_results_csv()
    save_results(results, results_csv)
```

## 优势

1. **集中管理**: 所有配置在一个地方,易于维护
2. **避免硬编码**: 不再有魔数和硬编码路径
3. **易于修改**: 修改一处,全局生效
4. **向后兼容**: 支持旧代码路径
5. **类型提示**: 更好的IDE支持
6. **自动创建目录**: 路径不存在时自动创建
