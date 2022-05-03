# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl
import numpy as np
import os 
import sys
import argparse
import itertools
from copy import deepcopy

import torch
from torch.optim import Adam

from models import ActorCriticNetwork 
from buffer import ReplayBuffer 

from torch.utils.tensorboard import SummaryWriter

def test(args):   

    num_test_episodes = 100
    device = torch.device("cuda")
    """
    ### Initialize the environment
    """
    env = gym.make('cruise-ctrl-v0', train=False, noise_required=False) 
    state_space = env.observation_space
    action_space = env.action_space 

    """
    ### Instantiating the neural network policies and value function estimators 
    ### We have an actor critic and a target actor critic which helps in reducing the maximization bias problem
    """
    ActorCritic = ActorCriticNetwork(state_space, action_space, device) 
    ActorCritic.load_state_dict(torch.load(os.path.join(args.log_dir, 'own_sac_best_model.pt')))
    ActorCritic.eval()

    def get_action(state, deterministic=False):
        return ActorCritic.act(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic) 

    def test_agent(test_env): 
        random_seeds = np.random.choice(10000, size=(num_test_episodes,))
        
        test_episodes_returns = []
        for test_ep_idx in range(num_test_episodes):
            state = test_env.reset(seed=random_seeds[test_ep_idx])
            done = False 
            episode_return = 0

            while not done: 
                state_tensor = torch.tensor(state).unsqueeze(0)
                state, reward, done, _ = test_env.step(get_action(state_tensor, deterministic=True).squeeze(0))
                episode_return += reward

            test_episodes_returns.append(episode_return)

        return test_episodes_returns 

    test_agent(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/tmp/sac/sac_for_basic", help="logging directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    test(args)