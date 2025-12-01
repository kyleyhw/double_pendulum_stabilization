import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Learnable Log Standard Deviation for continuous action
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic Network (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Returns mean action and state value
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value
    
    def get_action(self, state):
        # Sample action from policy
        action_mean = self.actor(state)
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        # Evaluate action log_prob and state value, and entropy
        action_mean = self.actor(state)
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        
        return log_prob, value, entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, log_prob = self.policy.get_action(state)
        return action.cpu().numpy(), log_prob.cpu().numpy()
    
    def update(self, memory):
        # Unpack memory
        states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(memory.rewards)).to(self.device)
        is_terminals = torch.FloatTensor(np.array(memory.is_terminals)).to(self.device)
        
        # Monte Carlo Estimate of State Rewards
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device)
        # Normalize rewards
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-5)
        
        # PPO Update Loop
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = torch.squeeze(state_values)
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_log_probs)
            
            # Surrogate Loss
            advantages = rewards_to_go - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Total Loss = -min(surr1, surr2) + 0.5*MSE(v, R) - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss_fn(state_values, rewards_to_go) - 0.01 * dist_entropy
            
            # Gradient Step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
        
    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
