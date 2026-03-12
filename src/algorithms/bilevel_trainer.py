"""
双层优化训练器,用于对抗入侵检测系统

本模块实现了真正的双层优化(Stackelberg博弈):
    - 内层循环(跟随者): 攻击者针对固定的防御者进行优化
    - 外层循环(领导者): 防御者针对最佳响应的攻击者进行优化

参考文献:
    Liu et al. "DARTS: Differentiable Architecture Search" ICLR 2019
    Colson et al. "An overview of bilevel optimization" Annals of OR 2007
"""

import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, Optional

from src.utils.config import Config
from src.utils.ppo import PPO
from src.agents.attacker_agent import AttackerAgent
from src.agents.defender_agent import DefenderAgent
from src.envs.network_security_game import NetworkSecurityGame

# 可选的 TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # 类型提示兼容


class BiLevelTrainer:
    """
    真正的双层对抗强化学习训练器
    
    架构:
        领导者(外层循环): 防御者
        跟随者(内层循环): 攻击者
        
    训练流程:
        1. 冻结防御者参数 θ_D
        2. 训练攻击者直到收敛(或K_inner步) → θ_A*
        3. 固定攻击者于最佳响应 θ_A*
        4. 针对θ_A*训练防御者1步 → 更新θ_D
        5. 重复
        
    这与简单的多智能体RL不同(后者同时更新两个智能体)。
    """
    
    def __init__(
        self,
        env: NetworkSecurityGame,
        attacker: AttackerAgent,
        defender: DefenderAgent,
        ppo_attacker: PPO,
        ppo_defender: PPO,
        config: Config,
        logger: Optional[SummaryWriter] = None
    ):
        """
        初始化双层训练器
        
        参数:
            env: 对抗游戏环境
            attacker: 攻击者智能体(跟随者)
            defender: 防御者智能体(领导者)
            ppo_attacker: 攻击者的PPO包装器
            ppo_defender: 防御者的PPO包装器
            config: 训练配置
            logger: TensorBoard日志记录器(可选)
        """
        self.env = env
        self.attacker = attacker
        self.defender = defender
        self.ppo_attacker = ppo_attacker
        self.ppo_defender = ppo_defender
        self.config = config
        self.logger = logger
        
        # Bi-level specific hyperparameters
        self.inner_loop_steps = config.INNER_LOOP_STEPS if hasattr(config, 'INNER_LOOP_STEPS') else 5
        self.kl_threshold = config.KL_THRESHOLD if hasattr(config, 'KL_THRESHOLD') else 0.01
        
        # ==================== 内层循环稳定化参数 ====================
        # 攻击者熵正则化退火调度
        self.entropy_coeff = getattr(config, 'ENTROPY_COEFF_INIT', 0.05)
        self.entropy_coeff_min = getattr(config, 'ENTROPY_COEFF_MIN', 0.001)
        self.entropy_decay = getattr(config, 'ENTROPY_DECAY', 0.995)
        # 攻击者热身调度
        self.warmup_ratio = getattr(config, 'WARMUP_RATIO', 0.2)
        self.warmup_kinner = getattr(config, 'WARMUP_KINNER', 1)
        self.total_episodes = getattr(config, 'RL_EPISODES', 100)
        
        # 将初始熵系数应用到攻击者 PPO
        self.ppo_attacker.set_entropy_coeff(self.entropy_coeff)
        
        # Metrics tracking
        self.global_step = 0
        self.inner_loop_converged_count = 0
        
    def collect_trajectory(
        self, 
        agent_type: str,
        max_steps: int = 50
    ) -> Tuple[float, Dict[str, float]]:
        """
        Collect one episode trajectory for specified agent.
        
        Args:
            agent_type: "attacker" or "defender"
            max_steps: Maximum steps per episode
            
        Returns:
            episode_reward: Total reward for this episode
            info: Additional statistics
        """
        state, _ = self.env.reset()
        episode_reward = 0
        step_count = 0
        
        terminated = False
        while not terminated and step_count < max_steps:
            # 攻击者先在原始状态上决策,防御者在扰动后的观测上决策。
            action_att, logprob_att = self.attacker.get_action(state)
            defender_state = self.env.get_defender_observation(action_att, state=state)
            action_def, logprob_def = self.defender.get_action(defender_state)
            
            # Store experience in appropriate buffer
            if agent_type == "attacker":
                self.ppo_attacker.buffer.states.append(torch.FloatTensor(state).to(Config.DEVICE))
                self.ppo_attacker.buffer.actions.append(torch.tensor(action_att).to(Config.DEVICE))
                self.ppo_attacker.buffer.logprobs.append(logprob_att)
            else:  # defender
                self.ppo_defender.buffer.states.append(torch.FloatTensor(defender_state).to(Config.DEVICE))
                self.ppo_defender.buffer.actions.append(torch.tensor(action_def).to(Config.DEVICE))
                self.ppo_defender.buffer.logprobs.append(logprob_def)
            
            # Environment step
            actions = {
                'attacker': action_att,
                'defender': action_def,
                'modified_state': defender_state
            }
            next_state, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Store reward and terminal flag
            if agent_type == "attacker":
                self.ppo_attacker.buffer.rewards.append(rewards['attacker'])
                self.ppo_attacker.buffer.is_terminals.append(terminated)
                episode_reward += rewards['attacker']
            else:
                self.ppo_defender.buffer.rewards.append(rewards['defender'])
                self.ppo_defender.buffer.is_terminals.append(terminated)
                episode_reward += rewards['defender']
            
            state = next_state
            step_count += 1
            
        return episode_reward, {'steps': step_count, 'success': info.get('is_success', False)}
    
    def check_inner_convergence(self) -> bool:
        """
        Check if Attacker (Inner Loop) has converged.
        
        Convergence Criterion:
            KL divergence between old and new policy < threshold
            
        Returns:
            True if converged, False otherwise
        """
        if len(self.ppo_attacker.buffer.states) < 10:
            return False  # Not enough data
            
        # Compute approximate KL divergence
        with torch.no_grad():
            old_states = torch.stack(self.ppo_attacker.buffer.states[-10:])
            old_actions = torch.stack(self.ppo_attacker.buffer.actions[-10:])
            old_logprobs = torch.stack(self.ppo_attacker.buffer.logprobs[-10:])
            
            # Get new log probs
            new_logprobs, _, _ = self.attacker.evaluate(old_states, old_actions)
            
            # KL(old || new) ≈ E[log(old) - log(new)]
            approx_kl = (old_logprobs - new_logprobs).mean().item()
            
        return abs(approx_kl) < self.kl_threshold
    
    def train_inner_loop(self, episode: int) -> Dict[str, float]:
        """
        Inner Loop: Train Attacker to best-response against fixed Defender.
        
        This is the FOLLOWER optimization:
            max_{θ_A} E[R_A(s, a_D^fixed, a_A; θ_A)]
            
        Args:
            episode: Current episode number
            
        Returns:
            Metrics dictionary
        """
        # Freeze Defender parameters
        defender_frozen_state = deepcopy(self.defender.state_dict())
        self.defender.eval()  # Set to eval mode to freeze BatchNorm etc.
        
        inner_rewards = []
        converged_at_step = None
        
        # 根据热身调度确定当前 episode 的实际内层步数
        effective_steps = self._get_effective_inner_steps(episode)
        
        for inner_step in range(effective_steps):
            # Collect trajectory
            reward, info = self.collect_trajectory(agent_type="attacker")
            inner_rewards.append(reward)
            
            # Update Attacker
            if len(self.ppo_attacker.buffer.states) > 0:
                self.ppo_attacker.update()
                
            # Check convergence
            if self.check_inner_convergence():
                converged_at_step = inner_step + 1
                self.inner_loop_converged_count += 1
                print(f"  [Inner Loop] Converged at step {converged_at_step}/{self.inner_loop_steps}")
                break
        
        # Log metrics
        avg_inner_reward = np.mean(inner_rewards) if inner_rewards else 0.0
        
        # 计算攻击者当前策略熵（诊断指标）
        attacker_entropy = self._compute_attacker_entropy()
        
        if self.logger:
            self.logger.add_scalar('InnerLoop/AvgReward', avg_inner_reward, episode)
            self.logger.add_scalar('InnerLoop/ConvergedAt', converged_at_step or effective_steps, episode)
            self.logger.add_scalar('InnerLoop/AttackerEntropy', attacker_entropy, episode)
            self.logger.add_scalar('InnerLoop/EntropyCoeff', self.entropy_coeff, episode)
            self.logger.add_scalar('InnerLoop/EffectiveKInner', effective_steps, episode)
            
        return {
            'inner_avg_reward': avg_inner_reward,
            'inner_steps': converged_at_step or effective_steps,
            'inner_converged': converged_at_step is not None,
            'attacker_entropy': attacker_entropy,
            'entropy_coeff': self.entropy_coeff,
            'effective_k_inner': effective_steps
        }
    
    def train_outer_loop(self, episode: int) -> Dict[str, float]:
        """
        Outer Loop: Train Defender against best-response Attacker.
        
        This is the LEADER optimization:
            max_{θ_D} E[R_D(s, a_D, a_A^*; θ_D)]
            where a_A^* is the best-response from Inner Loop
            
        Args:
            episode: Current episode number
            
        Returns:
            Metrics dictionary
        """
        # Fix Attacker at best-response (freeze parameters)
        self.attacker.eval()
        self.attacker.requires_grad_(False)
        
        # Unfreeze Defender
        self.defender.train()
        self.defender.requires_grad_(True)
        
        # Collect trajectory for Defender
        reward, info = self.collect_trajectory(agent_type="defender")
        
        # Update Defender
        if len(self.ppo_defender.buffer.states) > 0:
            self.ppo_defender.update()
        
        # Re-enable Attacker gradients for next Inner Loop
        self.attacker.requires_grad_(True)
        
        # Log metrics
        if self.logger:
            self.logger.add_scalar('OuterLoop/Reward', reward, episode)
            
        return {
            'outer_reward': reward,
            'outer_steps': info['steps']
        }
    
    def train_one_episode(self, episode: int) -> Dict[str, float]:
        """
        Complete Bi-level Training for One Episode.
        
        Flow:
            1. Inner Loop: Train Attacker to convergence
            2. Outer Loop: Train Defender against best-response Attacker
            
        Args:
            episode: Current episode number
            
        Returns:
            Combined metrics from both loops
        """
        print(f"\n=== Episode {episode} ===")
        
        # Phase 1: Inner Loop (Follower)
        print("  [Phase 1] Inner Loop: Training Attacker...")
        inner_metrics = self.train_inner_loop(episode)
        
        # Phase 2: Outer Loop (Leader)
        print("  [Phase 2] Outer Loop: Training Defender...")
        outer_metrics = self.train_outer_loop(episode)
        
        # Combine metrics
        metrics = {**inner_metrics, **outer_metrics}
        
        # 更新熵正则化退火调度
        self._update_entropy_schedule(episode)
        
        # Log combined metrics
        if self.logger:
            self.logger.add_scalar('Training/InnerOuterRewardRatio', 
                                   metrics['inner_avg_reward'] / (metrics['outer_reward'] + 1e-6), 
                                   episode)
        
        self.global_step += 1
        
        return metrics
    
    def save_checkpoint(self, save_dir: str, episode: int):
        """Save model checkpoints."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'episode': episode,
            'attacker_state_dict': self.attacker.state_dict(),
            'defender_state_dict': self.defender.state_dict(),
            'ppo_attacker_optimizer': self.ppo_attacker.optimizer.state_dict(),
            'ppo_defender_optimizer': self.ppo_defender.optimizer.state_dict(),
            'inner_loop_converged_count': self.inner_loop_converged_count
        }, os.path.join(save_dir, f'checkpoint_ep{episode}.pth'))
        
        print(f"Checkpoint saved at episode {episode}")
    
    # ==================== 内层循环稳定化方法 ====================
    
    def _get_effective_inner_steps(self, episode: int) -> int:
        """
        攻击者热身调度：训练初期限制内层步数，防止攻击者过度优化。
        
        调度策略：
            - 前 warmup_ratio 的 episode: K_inner = warmup_kinner (=1)
            - 之后线性过渡到配置的 inner_loop_steps
        
        Args:
            episode: 当前 episode 编号
            
        Returns:
            当前 episode 应用的内层循环步数
        """
        warmup_episodes = int(self.total_episodes * self.warmup_ratio)
        
        if episode < warmup_episodes:
            # 热身期：使用较小的 K_inner
            return self.warmup_kinner
        
        # 热身后：线性过渡到完整的 inner_loop_steps
        progress = min(1.0, (episode - warmup_episodes) / max(1, warmup_episodes))
        effective = self.warmup_kinner + int(progress * (self.inner_loop_steps - self.warmup_kinner))
        return min(effective, self.inner_loop_steps)
    
    def _update_entropy_schedule(self, episode: int):
        """
        熵正则化退火调度：每个 episode 后指数衰减攻击者熵系数。
        
        目的：
            - 训练初期：高熵系数鼓励探索，防止攻击者策略过早坍缩
            - 训练后期：低熵系数允许攻击者收敛到确定性策略
        """
        self.entropy_coeff = max(
            self.entropy_coeff_min,
            self.entropy_coeff * self.entropy_decay
        )
        # 同步更新攻击者 PPO 的熵系数
        self.ppo_attacker.set_entropy_coeff(self.entropy_coeff)
    
    def _compute_attacker_entropy(self) -> float:
        """
        计算攻击者当前策略的平均熵（诊断指标）。
        
        低熵 → 策略坍缩（可能过度优化）
        高熵 → 策略随机（未学到有效攻击）
        
        Returns:
            攻击者策略的平均熵值
        """
        if len(self.ppo_attacker.buffer.states) < 5:
            return 0.0
        
        with torch.no_grad():
            states = torch.stack(self.ppo_attacker.buffer.states[-5:])
            actions = torch.stack(self.ppo_attacker.buffer.actions[-5:])
            _, _, entropy = self.attacker.evaluate(states, actions)
            return entropy.mean().item()

