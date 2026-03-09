"""
Ablation Variant: Bi-level WITHOUT Inner Loop Convergence

关键区别:
- Attacker使用固定的随机策略(不训练)
- 无KL散度收敛检测
- Defender仍然训练(Outer Loop)

目的: 证明Inner Loop收敛的价值
"""

import numpy as np
import torch
import sys
import os
import argparse

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # baselines -> src -> project_root
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.attacker_agent import AttackerAgent
from src.agents.defender_agent import DefenderAgent
from src.envs.network_security_game import NetworkSecurityGame
from src.utils.config import Config
from src.utils.ppo import PPO

# 可选的 TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def train_fixed_attacker(seed=42, num_episodes=100, log_dir='runs/ablation_fixed_att'):
    """
    Ablation: 固定随机Attacker训练
    
    参数:
        seed: 随机种子
        num_episodes: 训练轮数
        log_dir: TensorBoard 日志目录
    """
    # 设置随机种子
    Config.set_seed(seed)
    
    # 初始化日志记录器
    logger = SummaryWriter(f'{log_dir}_seed{seed}') if TENSORBOARD_AVAILABLE else None
    
    # 初始化环境
    env = NetworkSecurityGame()
    
    # 初始化Agents
    attacker_net = AttackerAgent().to(Config.DEVICE)
    defender_net = DefenderAgent().to(Config.DEVICE)
    
    # 🔴 关键: Attacker不训练,使用固定初始化参数
    # 冻结Attacker参数
    for param in attacker_net.parameters():
        param.requires_grad = False
    
    # 只为Defender创建优化器
    defender_optimizer = torch.optim.Adam(defender_net.parameters(), lr=Config.LR)
    
    # 只为Defender创建PPO
    ppo_defender = PPO(
        defender_net, defender_optimizer,
        Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS
    )
    
    print(f"\n{'='*60}")
    print(f"Ablation: w/o Inner Loop Convergence")
    print(f"{'='*60}")
    print(f"设备: {Config.DEVICE}")
    print(f"随机种子: {seed}")
    print(f"训练轮数: {num_episodes}")
    print(f"Attacker: 固定随机策略 (不训练)")
    print(f"Defender: 正常训练")
    print(f"{'='*60}\n")
    
    # 训练循环
    for episode in range(1, num_episodes + 1):
        print(f"\n=== Episode {episode} ===")
        
        # 每个episode收集经验
        state, _ = env.reset()
        episode_reward_def = 0
        
        for step in range(Config.MAX_STEPS):
            # 🔴 Attacker使用固定随机策略(不更新)
            action_att, _ = attacker_net.get_action(state)
            defender_state = env.get_defender_observation(action_att, state=state)

            # Defender基于扰动后的状态决策
            action_def, logprob_def = defender_net.get_action(defender_state)
            
            # 存储Defender的experience
            ppo_defender.buffer.states.append(torch.FloatTensor(defender_state).to(Config.DEVICE))
            ppo_defender.buffer.actions.append(torch.tensor(action_def).to(Config.DEVICE))
            ppo_defender.buffer.logprobs.append(logprob_def)
            
            # 环境step
            actions = {
                'attacker': action_att,
                'defender': action_def,
                'modified_state': defender_state
            }
            next_state, rewards, terminated, truncated, info = env.step(actions)
            
            # 存储Defender reward
            reward_def = rewards['defender']
            ppo_defender.buffer.rewards.append(reward_def)
            ppo_defender.buffer.is_terminals.append(terminated)
            
            episode_reward_def += reward_def
            
            state = next_state
            
            if terminated:
                break
        
        # Outer Loop: 更新Defender (Attacker不更新)
        if len(ppo_defender.buffer.states) > 0:
            ppo_defender.update()
        
        # 记录日志
        if logger:
            logger.add_scalar('Training/DefenderReward', episode_reward_def, episode)
        
        # 打印进度  
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Defender奖励 = {episode_reward_def:.3f}")
        
        # 保存检查点
        if episode % 50 == 0:
            ckpt_path = Config.get_checkpoint_path("FixedAttacker", seed, episode)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'episode': episode,
                'attacker_state_dict': attacker_net.state_dict(),
                'defender_state_dict': defender_net.state_dict(),
                'defender_optimizer': defender_optimizer.state_dict(),
            }, ckpt_path)
    
    # 保存最终模型
    attacker_path = Config.get_model_path("FixedAttacker", seed, "attacker")
    defender_path = Config.get_model_path("FixedAttacker", seed, "defender")
    
    torch.save(attacker_net.state_dict(), attacker_path)
    torch.save(defender_net.state_dict(), defender_path)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"Attacker(固定): {attacker_path.relative_to(Config.PROJECT_ROOT)}")
    print(f"Defender模型: {defender_path.relative_to(Config.PROJECT_ROOT)}")
    print(f"{'='*60}\n")
    
    if logger:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation: w/o Inner Loop')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--log_dir', type=str, default='runs/ablation_fixed_att', 
                       help='TensorBoard日志目录')
    parser.add_argument('--dataset', type=str, default='nsl-kdd', help='数据集名称: nsl-kdd / unsw-nb15')
    
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)
    
    train_fixed_attacker(seed=args.seed, num_episodes=args.episodes, log_dir=args.log_dir)
