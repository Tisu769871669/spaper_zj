
import numpy as np
import torch
import sys
import os
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (src 的上一级)
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from src.envs.network_security_game import NetworkSecurityGame
from src.agents.attacker_agent import AttackerAgent
from src.agents.defender_agent import DefenderAgent
from src.utils.config import Config
from src.utils.ppo import PPO

import argparse

def train(seed=42, mode='full'):
    Config.set_seed(seed)
    print(f"Initializing environment (Mode: {mode}, Seed: {seed})...")
    env = NetworkSecurityGame()
    
    # 初始化 PPO 智能体
    attacker_net = AttackerAgent().to(Config.DEVICE)
    defender_net = DefenderAgent().to(Config.DEVICE)
    
    # 优化器
    optimizer_att = torch.optim.Adam(attacker_net.parameters(), lr=Config.LR)
    optimizer_def = torch.optim.Adam(defender_net.parameters(), lr=Config.LR)
    
    # PPO 包装器
    ppo_attacker = PPO(attacker_net, optimizer_att, Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS)
    ppo_defender = PPO(defender_net, optimizer_def, Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS)
    
    print(f"Starting Training on {Config.DEVICE}...")
    
    # 简单的循环验证仿真
    for episode in range(1, Config.K_EPOCHS + 5): # 稍微多跑几轮验证 Update
        state, _ = env.reset()
        episode_reward_att = 0
        episode_reward_def = 0
        
        terminated = False
        while not terminated:
            # PPO Select Actions (带梯度追踪的采样)
            action_att, logprob_att = attacker_net.get_action(state)
            defender_state = env.get_defender_observation(action_att, state=state)
            action_def, logprob_def = defender_net.get_action(defender_state)
            
            # --- Ablation Logic ---
            if mode == 'defender_only':
                # In ablation mode, Attacker does not learn or effectively impact
                # We can simulate this by not storing its experiences (so no update)
                # But for consistency in 'step', we keep the action.
                pass

            # 保存到缓冲区
            ppo_attacker.buffer.states.append(torch.FloatTensor(state).to(Config.DEVICE))
            ppo_attacker.buffer.actions.append(torch.tensor(action_att).to(Config.DEVICE))
            ppo_attacker.buffer.logprobs.append(logprob_att)
            
            ppo_defender.buffer.states.append(torch.FloatTensor(defender_state).to(Config.DEVICE))
            ppo_defender.buffer.actions.append(torch.tensor(action_def).to(Config.DEVICE))
            ppo_defender.buffer.logprobs.append(logprob_def)
            
            actions = {
                'attacker': action_att,
                'defender': action_def,
                'modified_state': defender_state
            }
            
            # Step
            next_state, rewards, terminated, truncated, info = env.step(actions)
            
            # 保存 Reward 和 Terminal
            ppo_attacker.buffer.rewards.append(rewards['attacker'])
            ppo_attacker.buffer.is_terminals.append(terminated)
            
            ppo_defender.buffer.rewards.append(rewards['defender'])
            ppo_defender.buffer.is_terminals.append(terminated)
            
            episode_reward_att += rewards['attacker']
            episode_reward_def += rewards['defender']
            
            state = next_state
        
        # PPO Update
        if episode % 10 == 0:
            if mode == 'full':
                ppo_attacker.update()
                ppo_defender.update()
            elif mode == 'defender_only':
                ppo_defender.update()
                ppo_attacker.buffer.clear() # Clear but don't update attacker
            
        # print(f"Episode {episode}: ...")
    
    # 保存模型
    torch.save(attacker_net.state_dict(), os.path.join(project_root, "src", "attacker.pth"))
    torch.save(defender_net.state_dict(), os.path.join(project_root, "src", "defender.pth"))
    print("Models saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default='full', choices=['full', 'defender_only'])
    parser.add_argument("--dataset", type=str, default='nsl-kdd')
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)
    
    train(seed=args.seed, mode=args.mode)
