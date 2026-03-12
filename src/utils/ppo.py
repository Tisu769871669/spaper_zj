
import torch
import torch.nn as nn
from src.utils.config import Config

class RolloutBuffer:
    """
    经验回放缓冲区，用于存储 PPO 训练所需的轨迹数据。
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    """
    PPO (Proximal Policy Optimization) 算法实现。
    
    支持可配置的熵系数，用于内层循环稳定化。
    """
    def __init__(self, policy_net, optimizer, lr, gamma, eps_clip, k_epochs, entropy_coeff=0.01):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff  # 可动态调整的熵系数
        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()
    
    def set_entropy_coeff(self, coeff: float):
        """动态设置熵系数（用于内层循环退火调度）。"""
        self.entropy_coeff = coeff

    def update(self):
        """
        使用缓冲区中的数据更新策略网络。
        """
        # 1. 计算蒙特卡洛回报 (Monte Carlo Estimate of Returns)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 归一化回报 (可选，但推荐)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(Config.DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 将列表转换为张量
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(Config.DEVICE)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(Config.DEVICE)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(Config.DEVICE)
        
        # K 次 PPO 更新
        for _ in range(self.k_epochs):
            # 评估旧状态和动作
            # 注意：这里假设 policy_net 有一个 evaluate 方法，或者我们需要修改 Agent 结构
            logprobs, state_values, dist_entropy = self.policy_net.evaluate(old_states, old_actions)
            
            # 计算比率 (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算 Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 最终 Loss = -min(surr1, surr2) + 0.5 * MSE(value, reward) - entropy_coeff * entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - self.entropy_coeff * dist_entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 清空缓冲区
        self.buffer.clear()
