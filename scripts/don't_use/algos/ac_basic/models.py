import numpy as np
#import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20 

"""
### Actor Network class
"""
class ActorNetwork(nn.Module):

    def __init__(self, state_space, action_space, max_action, device):
        super(ActorNetwork, self).__init__() 

        self.state_dim = state_space.shape[0] 
        self.action_dim = action_space.shape[0] 

        #print(self.state_dim, self.action_dim)

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu_layer = nn.Linear(256, self.action_dim)
        self.fc_log_std_layer = nn.Linear(256, self.action_dim)

        self.max_action = max_action
        #self.max_action = self.max_action.to(device)

    def forward(self, state, deterministic=False):
        # state should be a batch_size x state_dim tensor 
        
        out = self.fc1(state)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        mu = self.fc_mu_layer(out) 
        log_std = self.fc_log_std_layer(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # print(mu.shape, std.shape)

        action_distribution = Normal(mu, std)
        if deterministic:
            sample_action = mu
        else:
            sample_action = action_distribution.rsample() 
        
        #print(sample_action.shape, "----")
        #print(action_distribution, "----")

        logprob_action = action_distribution.log_prob(sample_action).sum(axis=-1)
        logprob_action -= (2*(np.log(2) - sample_action - F.softplus(-2*sample_action))).sum(axis=1)
        action = torch.tanh(sample_action) * self.max_action

        # Returned actions need to be in batch_size x action_dim shape 
        # Returned logprob_action needs to be in batch_size shape
        return action, logprob_action


"""
### Critic Network class
"""
class CriticNetwork(nn.Module):

    def __init__(self, state_space, action_space):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_space.shape[0] 
        self.action_dim = action_space.shape[0] 

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # state should be a batch_size x state_dim tensor 
        # action should be a batch_size x action_dim tensor
        state_action_value = self.fc1(torch.cat([state, action], dim=1))
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc2(state_action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc3(state_action_value)

        return state_action_value.squeeze(-1) # returns batch_size x 1 tensor

"""
### Actor Critic Network class
"""
class ActorCriticNetwork(nn.Module):

    def __init__(self, state_space, action_space, device):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.max_action = self.action_space.high[0]

        # build policy and value functions
        self.policy = ActorNetwork(self.state_space, self.action_space, self.max_action, device)
        self.critic = CriticNetwork(self.state_space, self.action_space)

    def act(self, state, deterministic=False):
        with torch.no_grad():
            action, logprob_action = self.policy(state, deterministic=deterministic)
            return action.numpy(), logprob_action
