"""
训练输出目录结构管理器

统一管理所有训练输出文件,包括模型、日志、结果等
创建清晰的目录结构,便于查找和管理
"""

import os
from pathlib import Path
from datetime import datetime


class OutputManager:
    """
    输出目录管理器
    
    统一的目录结构:
        outputs/
        ├── models/              # 训练好的模型
        │   ├── BiARL/
        │   │   ├── seed42/
        │   │   │   ├── attacker.pth
        │   │   │   └── defender.pth
        │   │   ├── seed101/
        │   │   └── seed202/
        │   ├── VanillaPPO/
        │   └── LSTM/
        ├── logs/                # 训练日志
        │   ├── BiARL/
        │   │   ├── seed42.json
        │   │   ├── seed101.json
        │   │   └── seed202.json
        │   ├── VanillaPPO/
        │   └── LSTM/
        ├── results/             # 评估结果
        │   ├── experiment_results.csv
        │   ├── statistical_tests.json
        │   └── figures/         # 图表
        │       ├── recall_comparison.png
        │       └── training_curves.png
        ├── checkpoints/         # 训练检查点
        │   ├── BiARL/
        │   ├── VanillaPPO/
        │   └── LSTM/
        └── README.md           # 目录说明
    """
    
    def __init__(self, base_dir: str = "outputs"):
        """
        初始化输出管理器
        
        参数:
            base_dir: 基础输出目录,默认为 "outputs"
        """
        self.base_dir = Path(base_dir)
        self._create_structure()
    
    def _create_structure(self):
        """创建输出目录结构"""
        # 主要目录
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.figures_dir = self.results_dir / "figures"
        
        # 创建所有目录
        for directory in [
            self.models_dir,
            self.logs_dir,
            self.results_dir,
            self.checkpoints_dir,
            self.figures_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 为每个模型创建子目录
        model_types = ["BiARL", "VanillaPPO", "LSTM", "RandomForest"]
        for model_type in model_types:
            (self.models_dir / model_type).mkdir(exist_ok=True)
            (self.logs_dir / model_type).mkdir(exist_ok=True)
            (self.checkpoints_dir / model_type).mkdir(exist_ok=True)
    
    def get_model_path(self, model_type: str, seed: int, model_name: str = None) -> Path:
        """
        获取模型保存路径
        
        参数:
            model_type: 模型类型,如 "BiARL", "VanillaPPO"
            seed: 随机种子
            model_name: 模型名称,如 "attacker", "defender"
            
        返回:
            模型文件路径
        """
        model_dir = self.models_dir / model_type / f"seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name:
            return model_dir / f"{model_name}.pth"
        else:
            return model_dir / "model.pth"
    
    def get_log_path(self, model_type: str, seed: int) -> Path:
        """
        获取训练日志路径
        
        参数:
            model_type: 模型类型
            seed: 随机种子
            
        返回:
            日志文件路径
        """
        return self.logs_dir / model_type / f"seed{seed}.json"
    
    def get_checkpoint_path(self, model_type: str, seed: int, episode: int) -> Path:
        """
        获取检查点保存路径
        
        参数:
            model_type: 模型类型
            seed: 随机种子
            episode: episode编号
            
        返回:
            检查点文件路径
        """
        ckpt_dir = self.checkpoints_dir / model_type / f"seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"checkpoint_ep{episode}.pth"
    
    def get_results_csv_path(self) -> Path:
        """获取实验结果CSV路径"""
        return self.results_dir / "experiment_results.csv"
    
    def get_figure_path(self, figure_name: str) -> Path:
        """
        获取图表保存路径
        
        参数:
            figure_name: 图表名称,如 "recall_comparison.png"
            
        返回:
            图表文件路径
        """
        return self.figures_dir / figure_name
    
    def create_readme(self):
        """创建目录说明文件"""
        readme_content = f"""# 训练输出目录说明

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 目录结构

```
outputs/
├── models/              # 训练好的模型权重
│   ├── BiARL/
│   │   ├── seed42/
│   │   │   ├── attacker.pth    # 攻击者模型
│   │   │   └── defender.pth    # 防御者模型
│   │   ├── seed101/
│   │   └── seed202/
│   ├── VanillaPPO/
│   │   └── seed42/
│   │       └── model.pth
│   └── LSTM/
│       └── seed42/
│           └── model.pth
│
├── logs/                # 训练过程JSON日志
│   ├── BiARL/
│   │   ├── seed42.json        # 详细训练日志
│   │   ├── seed101.json
│   │   └── seed202.json
│   ├── VanillaPPO/
│   └── LSTM/
│
├── results/             # 评估结果和分析
│   ├── experiment_results.csv  # 所有模型的性能指标
│   ├── statistical_tests.json  # 统计检验结果
│   └── figures/                # 可视化图表
│       ├── recall_comparison.png
│       ├── training_curves.png
│       └── ablation_study.png
│
└── checkpoints/         # 训练检查点(每50 episodes)
    ├── BiARL/
    ├── VanillaPPO/
    └── LSTM/
```

## 📄 文件说明

### 模型文件 (models/)
- **格式**: PyTorch .pth 文件
- **内容**: 神经网络参数(state_dict)
- **用途**: 加载模型进行评估或部署
- **大小**: 约60-70KB

### 训练日志 (logs/)
- **格式**: JSON文件
- **内容**: 每个episode的详细指标、配置、训练时间
- **用途**: 分析训练过程、绘制曲线、调试
- **大小**: 约50KB

### 实验结果 (results/)
- **experiment_results.csv**: 所有模型在所有测试条件下的性能
- **statistical_tests.json**: 统计显著性检验结果
- **figures/**: 论文用图表

### 检查点 (checkpoints/)
- **格式**: PyTorch checkpoint文件
- **内容**: 模型+优化器状态
- **用途**: 恢复训练、中间状态分析
- **保存频率**: 每50 episodes

## 🔍 快速查找

### 查看某个模型的训练结果
```bash
# Bi-ARL seed 42的模型
outputs/models/BiARL/seed42/

# 对应的训练日志
outputs/logs/BiARL/seed42.json
```

### 查看所有模型的对比结果
```bash
outputs/results/experiment_results.csv
```

### 查看论文图表
```bash
outputs/results/figures/
```

## 📊 文件统计

- **总模型数**: 18个 (3 seeds × 6 models)
- **总日志数**: 9个 (3 seeds × 3 RL models)
- **总大小**: 约1.5MB

## ⚙️ 使用方法

### Python代码中获取路径
```python
from src.utils.output_manager import OutputManager

manager = OutputManager()

# 保存模型
model_path = manager.get_model_path("BiARL", seed=42, model_name="defender")
torch.save(model.state_dict(), model_path)

# 保存日志
log_path = manager.get_log_path("BiARL", seed=42)

# 保存结果CSV
results_path = manager.get_results_csv_path()
```

## 🧹 清理

删除所有输出但保留目录结构:
```bash
# Windows
rd /s /q outputs\\models outputs\\logs outputs\\checkpoints
mkdir outputs\\models outputs\\logs outputs\\checkpoints

# Linux/Mac
rm -rf outputs/models/* outputs/logs/* outputs/checkpoints/*
```
"""
        
        readme_path = self.base_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ 目录说明已创建: {readme_path}")
    
    def print_structure(self):
        """打印目录结构"""
        print(f"\n{'='*60}")
        print(f"  输出目录结构")
        print(f"{'='*60}")
        print(f"📁 {self.base_dir}/")
        print(f"  ├── 📁 models/        (训练好的模型)")
        print(f"  ├── 📁 logs/          (训练日志)")
        print(f"  ├── 📁 results/       (评估结果)")
        print(f"  │   └── 📁 figures/   (图表)")
        print(f"  ├── 📁 checkpoints/   (检查点)")
        print(f"  └── 📄 README.md      (说明文档)")
        print(f"{'='*60}\n")


# 初始化并创建目录结构
def setup_outputs(base_dir: str = "outputs"):
    """
    设置输出目录
    
    参数:
        base_dir: 输出基础目录
    """
    manager = OutputManager(base_dir)
    manager.create_readme()
    manager.print_structure()
    return manager


if __name__ == "__main__":
    # 创建输出目录结构
    manager = setup_outputs()
    
    # 示例: 获取路径
    print("\n示例路径:")
    print(f"Bi-ARL模型: {manager.get_model_path('BiARL', 42, 'defender')}")
    print(f"训练日志: {manager.get_log_path('BiARL', 42)}")
    print(f"结果CSV: {manager.get_results_csv_path()}")
    print(f"图表: {manager.get_figure_path('recall_comparison.png')}")
