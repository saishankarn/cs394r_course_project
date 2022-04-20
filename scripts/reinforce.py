from typing import Iterable
import numpy as np

import torch 
from torch.distributions import Categorical

torch.manual_seed(0)
 
class PolicyNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class PiApproximationWithNN():
    def __init__(self, 
                 state_dims,
                 num_actions,
                 alpha=1e-4):

        self.input_size = state_dims
        self.output_size = num_actions 
        self.PolicyNN = PolicyNet(self.input_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.PolicyNN.parameters(), lr=alpha, betas=(0.9, 0.999)) 
        self.PolicyNN.train()

    def __call__(self,s) -> int:
        # TODO: implement this method
        self.PolicyNN.eval()
        s = torch.Tensor(s).view(1, self.input_size)
        print(s)
        action_probs = self.PolicyNN(s)[0].detach().numpy()
        print(action_probs)
        action = np.random.choice(list(range(self.output_size)), p=action_probs)
        
        self.PolicyNN.train()

        return action
        #raise NotImplementedError()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method

        s = torch.Tensor(s).view(1, self.input_size)
        action_probs = self.PolicyNN(s)[0]
        
        action_choice = torch.Tensor(np.zeros((self.output_size,)))
        action_choice[a] = 1
        
        loss = -1*gamma_t*delta*torch.dot(torch.log(action_probs), action_choice)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Baseline(object):

    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class ValueNet(torch.nn.Module):
    def __init__(self, input_size):
        super(ValueNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):

        self.input_size = state_dims 
        self.ValueNN = ValueNet(self.input_size)
        self.optimizer = torch.optim.Adam(self.ValueNN.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.criterion = torch.nn.MSELoss()
        self.ValueNN.train()

    def __call__(self,s) -> float:
        # TODO: implement this method
        self.ValueNN.eval()
        s = torch.Tensor(s).view(1, self.input_size)
        state_value = self.ValueNN(s)
        self.ValueNN.train()
        return state_value.item()
        #raise NotImplementedError()

    def update(self,s,G):
        # TODO: implement this method
        s = torch.Tensor(s).view(1, self.input_size)
        
        V_old = self.ValueNN(s)
        label = torch.zeros(V_old.shape)
        label[0][0] = G
        loss = self.criterion(V_old, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    
    starting_returns = []
    for ep_idx in range(num_episodes):
        print(ep_idx)
        state, done = env.reset()
        episode = []
        while not done:
            action = pi(state)
            next_state, reward, done, = env.step(action)
            episode.append((state, action, next_state, reward))
            state = next_state
        
        returns = [] 
        G = 0
        for step in reversed(range(len(episode))):
            G = gamma*G + episode[step][-1]
            returns.append(G)
        returns.reverse()

        for step in range(len(episode)):
            state = episode[step][0]
            action = episode[step][1]
            gamma_t = pow(gamma, step)
            step_return = returns[step]
            delta = step_return - V(state)

            V.update(state, step_return)
            pi.update(state, action, gamma_t, delta)

        starting_returns.append(returns[0])
    return starting_returns

