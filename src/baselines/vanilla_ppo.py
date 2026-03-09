"""
Vanilla PPO Defender Baseline.

This is a NON-ADVERSARIAL baseline where the Defender is trained using standard PPO
without any attacker present. It serves to demonstrate the value of adversarial training
in the Bi-ARL framework.

Key Difference from Bi-ARL:
    - Bi-ARL: Defender trained against adaptive Attacker (worst-case scenarios)
    - Vanilla PPO: Defender trained on clean data only (no adversarial perturbations)
    
Expected Result:
    Vanilla PPO should achieve good performance on clean data but lower robustness
    under adversarial attacks compared to Bi-ARL.
"""

import torch
import numpy as np
import sys
import os
from typing import Dict

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agents.defender_agent import DefenderAgent
from src.utils.config import Config
from src.utils.ppo import PPO
from src.envs.network_security_game import NetworkSecurityGame


class VanillaPPOTrainer:
    """
    Trains a Defender using vanilla PPO without adversarial training.
    
    This serves as a baseline to show the importance of bi-level
    adversarial optimization in Bi-ARL.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize Vanilla PPO Trainer.
        
        Args:
            seed: Random seed for reproducibility
        """
        Config.set_seed(seed)
        self.seed = seed
        
        # Initialize environment (but will only use clean data, no attacker)
        self.env = NetworkSecurityGame()
        
        # Initialize Defender
        self.defender = DefenderAgent().to(Config.DEVICE)
        
        # Initialize optimizer and PPO
        self.optimizer = torch.optim.Adam(self.defender.parameters(), lr=Config.LR)
        self.ppo = PPO(
            self.defender, self.optimizer,
            Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS
        )
        
        print(f"\n{'='*60}")
        print(f"Vanilla PPO Defender Training")
        print(f"{'='*60}")
        print(f"Device: {Config.DEVICE}")
        print(f"Seed: {seed}")
        print(f"Mode: Single-Agent (NO Attacker)")
        print(f"{'='*60}\n")
    
    def collect_episode(self, max_steps: int = 50) -> Dict[str, float]:
        """
        Collect one episode of experience for Defender.
        
        Key Difference: No attacker actions - Defender sees CLEAN data only.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        state, _ = self.env.reset()
        episode_reward = 0
        correct_predictions = 0
        total_steps = 0
        
        terminated = False
        while not terminated and total_steps < max_steps:
            # Defender acts on CLEAN state (no adversarial perturbations)
            action_def, logprob_def = self.defender.get_action(state)
            
            # Store experience
            self.ppo.buffer.states.append(torch.FloatTensor(state).to(Config.DEVICE))
            self.ppo.buffer.actions.append(torch.tensor(action_def).to(Config.DEVICE))
            self.ppo.buffer.logprobs.append(logprob_def)
            
            # Environment step - NO ATTACKER ACTION
            # For vanilla training, we skip attacker perturbations
            actions = {
                'attacker': 0,  # Dummy action (no perturbation)
                'defender': action_def
            }
            
            next_state, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Only care about defender reward
            reward = rewards['defender']
            self.ppo.buffer.rewards.append(reward)
            self.ppo.buffer.is_terminals.append(terminated)
            
            episode_reward += reward
            if info.get('is_success', False):
                correct_predictions += 1
            
            state = next_state
            total_steps += 1
        
        return {
            'reward': episode_reward,
            'accuracy': correct_predictions / total_steps if total_steps > 0 else 0,
            'steps': total_steps
        }
    
    def train(self, num_episodes: int = 300) -> None:
        """
        Train Defender using vanilla PPO (no adversarial training).
        
        Args:
            num_episodes: Number of training episodes
        """
        print("Starting Vanilla PPO Training...\n")
        
        for episode in range(1, num_episodes + 1):
            # Collect episode
            stats = self.collect_episode()
            
            # Update policy every 10 episodes
            if episode % 10 == 0 and len(self.ppo.buffer.states) > 0:
                self.ppo.update()
            
            # Print progress
            if episode % 50 == 0:
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Reward: {stats['reward']:.2f}")
                print(f"  Accuracy: {stats['accuracy']:.2%}")
                print(f"  Steps: {stats['steps']}")
        
        # 保存模型(使用Config路径)
        save_path = Config.get_model_path("VanillaPPO", self.seed, "model")
        torch.save(self.defender.state_dict(), save_path)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Model saved to: {save_path.relative_to(Config.PROJECT_ROOT)}")
        print(f"{'='*60}\n")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vanilla PPO Defender Baseline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episodes', type=int, default=300, help='Number of training episodes')
    parser.add_argument('--dataset', type=str, default='nsl-kdd', help='Dataset: nsl-kdd / unsw-nb15')
    
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)
    
    trainer = VanillaPPOTrainer(seed=args.seed)
    trainer.train(num_episodes=args.episodes)


if __name__ == "__main__":
    main()
