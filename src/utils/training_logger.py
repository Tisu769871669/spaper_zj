"""
训练过程JSON日志记录器

记录每个episode的详细信息,包括奖励、损失、收敛状态等
输出为JSON格式,便于后续分析和可视化
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class TrainingLogger:
    """
    训练过程JSON日志记录器
    
    记录内容:
        - 训练配置(超参数、设备等)
        - 每个episode的指标(奖励、损失、收敛状态)
        - 训练时间统计
        - 最终模型路径
    """
    
    def __init__(self, log_dir: str, experiment_name: str, config: Dict[str, Any]):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志保存目录
            experiment_name: 实验名称(如 "bilevel_seed42")
            config: 训练配置字典
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{experiment_name}.json"
        
        # 初始化日志数据结构
        self.data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": config,
            "episodes": [],
            "summary": {}
        }
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """
        记录单个episode的指标
        
        参数:
            episode: episode编号
            metrics: 指标字典,如 {"inner_reward": 10.5, "outer_reward": -5.0}
        """
        episode_data = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.data["episodes"].append(episode_data)
    
    def log_summary(self, summary: Dict[str, Any]):
        """
        记录训练总结信息
        
        参数:
            summary: 总结字典,如 {"total_time": 600, "final_reward": 15.0}
        """
        self.data["summary"] = summary
        self.data["end_time"] = datetime.now().isoformat()
    
    def save(self):
        """保存日志到JSON文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 训练日志已保存: {self.log_file}")
    
    def load(self) -> Dict[str, Any]:
        """
        加载已有的日志文件
        
        返回:
            日志数据字典
        """
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}


class TrainingLoggerCollection:
    """
    多实验日志管理器
    
    用于管理多个seed、多个模型的训练日志
    """
    
    def __init__(self, base_dir: str = "logs"):
        """
        初始化日志集合管理器
        
        参数:
            base_dir: 基础日志目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储所有日志记录器
        self.loggers: Dict[str, TrainingLogger] = {}
    
    def create_logger(
        self,
        model_name: str,
        seed: int,
        config: Dict[str, Any]
    ) -> TrainingLogger:
        """
        创建新的日志记录器
        
        参数:
            model_name: 模型名称(如 "BiARL", "VanillaPPO")
            seed: 随机种子
            config: 训练配置
            
        返回:
            日志记录器实例
        """
        experiment_name = f"{model_name}_seed{seed}"
        logger = TrainingLogger(
            log_dir=str(self.base_dir / model_name),
            experiment_name=experiment_name,
            config=config
        )
        
        self.loggers[experiment_name] = logger
        return logger
    
    def save_all(self):
        """保存所有日志"""
        for logger in self.loggers.values():
            logger.save()
    
    def generate_summary_report(self, output_file: str = "training_summary.json"):
        """
        生成所有实验的汇总报告
        
        参数:
            output_file: 输出文件名
        """
        summary = {
            "total_experiments": len(self.loggers),
            "experiments": {}
        }
        
        for name, logger in self.loggers.items():
            if logger.data["episodes"]:
                summary["experiments"][name] = {
                    "total_episodes": len(logger.data["episodes"]),
                    "start_time": logger.data["start_time"],
                    "end_time": logger.data.get("end_time", "In progress"),
                    "config": logger.data["config"],
                    "summary": logger.data["summary"]
                }
        
        output_path = self.base_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 汇总报告已保存: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 示例: 创建日志记录器
    config = {
        "seed": 42,
        "lr": 3e-4,
        "gamma": 0.99,
        "episodes": 100
    }
    
    logger = TrainingLogger(
        log_dir="logs/BiARL",
        experiment_name="bilevel_seed42",
        config=config
    )
    
    # 模拟记录episode
    for ep in range(1, 6):
        metrics = {
            "inner_avg_reward": float(ep * 10),
            "outer_reward": float(-ep * 5),
            "inner_converged": ep % 2 == 0
        }
        logger.log_episode(ep, metrics)
    
    # 记录总结
    logger.log_summary({
        "total_time_seconds": 600,
        "convergence_rate": 0.4,
        "final_model_path": "src/defender_bilevel_seed42.pth"
    })
    
    # 保存
    logger.save()
    
    print("\n✅ 示例JSON日志已创建")
    print(f"查看文件: {logger.log_file}")
