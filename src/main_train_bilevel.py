"""
Bi-level 对抗强化学习主训练脚本

本版本实现了真正的双层优化。
如需对比标准的多智能体RL,请参考 main_train.py
"""

import numpy as np
import torch
import sys
import os
import argparse

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.attacker_agent import AttackerAgent
from src.agents.defender_agent import DefenderAgent
from src.envs.network_security_game import NetworkSecurityGame
from src.utils.config import Config
from src.utils.ppo import PPO
from src.algorithms.bilevel_trainer import BiLevelTrainer

# 可选的 TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard 未安装。训练将在无日志记录的情况下进行。")
    print("安装 TensorBoard: pip install tensorboard")


def train_bilevel(seed=42, num_episodes=300, log_dir='runs/bilevel'):
    """
    使用真正的双层优化进行训练
    
    参数:
        seed: 随机种子,用于结果可复现性
        num_episodes: 外层循环训练轮数
        log_dir: TensorBoard 日志目录
    """
    # 设置随机种子
    Config.set_seed(seed)
    
    # 初始化日志记录器(可选)
    logger = SummaryWriter(f'{log_dir}_seed{seed}') if TENSORBOARD_AVAILABLE else None
    
    print(f"\n{'='*60}")
    print(f"Bi-level 对抗强化学习训练")
    print(f"{'='*60}")
    print(f"设备: {Config.DEVICE}")
    print(f"随机种子: {seed}")
    print(f"训练轮数: {num_episodes}")
    print(f"内层循环步数: {Config.INNER_LOOP_STEPS}")
    print(f"KL 阈值: {Config.KL_THRESHOLD}")
    print(f"{'='*60}\n")
    
    # 初始化环境
    env = NetworkSecurityGame()
    
    # 初始化智能体
    attacker_net = AttackerAgent().to(Config.DEVICE)
    defender_net = DefenderAgent().to(Config.DEVICE)
    
    # 初始化优化器
    optimizer_att = torch.optim.Adam(attacker_net.parameters(), lr=Config.LR)
    optimizer_def = torch.optim.Adam(defender_net.parameters(), lr=Config.LR)
    
    # 初始化 PPO 包装器
    ppo_attacker = PPO(
        attacker_net, optimizer_att, 
        Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS
    )
    ppo_defender = PPO(
        defender_net, optimizer_def,
        Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS
    )
    
    # 初始化 Bi-level 训练器
    trainer = BiLevelTrainer(
        env=env,
        attacker=attacker_net,
        defender=defender_net,
        ppo_attacker=ppo_attacker,
        ppo_defender=ppo_defender,
        config=Config,
        logger=logger
    )
    
    # 训练循环
    for episode in range(1, num_episodes + 1):
        metrics = trainer.train_one_episode(episode)
        
        # 打印进度
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  内层循环: 平均奖励 = {metrics['inner_avg_reward']:.3f}, "
                  f"步数 = {metrics['inner_steps']}, 收敛 = {metrics['inner_converged']}")
            print(f"  外层循环: 奖励 = {metrics['outer_reward']:.3f}")
        
        # 保存检查点(使用Config路径)
        if episode % 50 == 0:
            ckpt_path = Config.get_checkpoint_path("BiARL", seed, episode)
            trainer.save_checkpoint(str(ckpt_path.parent), episode)
    
    # 保存最终模型(使用Config路径)
    attacker_path = Config.get_model_path("BiARL", seed, "attacker")
    defender_path = Config.get_model_path("BiARL", seed, "defender")
    
    torch.save(attacker_net.state_dict(), attacker_path)
    torch.save(defender_net.state_dict(), defender_path)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"Attacker模型已保存: {attacker_path.relative_to(Config.PROJECT_ROOT)}")
    print(f"Defender模型已保存: {defender_path.relative_to(Config.PROJECT_ROOT)}")
    print(f"TensorBoard 日志: {log_dir}_seed{seed}")
    print(f"内层循环收敛率: {trainer.inner_loop_converged_count}/{num_episodes} 轮")
    print(f"{'='*60}\n")
    
    if logger:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bi-level 对抗强化学习训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--episodes', type=int, default=300, help='训练轮数')
    parser.add_argument('--log_dir', type=str, default='runs/bilevel', help='TensorBoard 日志目录')
    parser.add_argument('--dataset', type=str, default='nsl-kdd', help='数据集名称: nsl-kdd / unsw-nb15')
    
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)
    
    train_bilevel(seed=args.seed, num_episodes=args.episodes, log_dir=args.log_dir)

