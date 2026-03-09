
import numpy as np
import torch
import torch.nn as nn
from src.utils.config import Config
from torch.distributions import Categorical

class DefenderAgent(nn.Module):
    def __init__(self):
        super(DefenderAgent, self).__init__()
        self.state_dim = Config.STATE_DIM
        self.action_dim = Config.ACTION_DIM_DEFENDER
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)
        state = state.to(Config.DEVICE)
        return self.actor(state)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(Config.DEVICE)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy
