# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl
import numpy as np
import os 
import sys
import argparse
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt 

import torch
from torch.optim import Adam

from models import ActorCriticNetwork 
from buffer import ReplayBuffer 

from torch.utils.tensorboard import SummaryWriter

def test(args):   

    num_test_episodes = 1
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
        random_seed = np.random.choice(10000)
        
        total_reward_list = [0]
        rel_dist_list = []
        fv_pos_list = []
        fv_vel_list = []
        fv_acc_list = []
        ego_pos_list = []
        ego_vel_list = []
        ego_acc_list = []

        state = test_env.reset(seed=random_seed)
        done = False 
        episode_return = 0

        while not done: 
            state_tensor = torch.tensor(state).unsqueeze(0)
            #print(state_tensor, state_tensor.shape)
            state, reward, done, info = test_env.step(get_action(state_tensor, deterministic=True).squeeze(0))
            episode_return += reward 

            # Gather results for plotting
            total_reward_list.append(total_reward_list[-1] + reward)
            rel_dist_list.append(state[0])

            fv_pos_list.append(info["fv_pos"])
            fv_vel_list.append(info["fv_vel"])
            fv_acc_list.append(info["fv_acc"])

            ego_pos_list.append(info["ego_pos"])
            ego_vel_list.append(info["ego_vel"])
            ego_acc_list.append(info["ego_acc"]) 

        """
        ### Generate Plots
        """

        fig, axes = plt.subplots(2,3, figsize=(15,7))
        plt.rcParams.update({'font.size': 10})

        axes[0, 0].plot(total_reward_list)
        axes[0, 1].plot(rel_dist_list)
        axes[1, 0].plot(fv_pos_list, color = 'b', label = 'Front vehicle')
        axes[1, 0].plot(ego_pos_list, color = 'r',  label = 'Ego vehicle')
        axes[1, 1].plot(fv_vel_list, color = 'b', label = 'Front vehicle')
        axes[1, 1].plot(ego_vel_list, color = 'r',  label = 'Ego vehicle')
        axes[1, 2].plot(fv_acc_list, color = 'b', label = 'Front vehicle')
        axes[1, 2].plot(ego_acc_list, color = 'r',  label = 'Ego vehicle')

        axes[0, 0].title.set_text('Total reward accumulated over time')
        axes[0, 1].title.set_text('Distance between vehicles over time')
        axes[1, 0].title.set_text('Position of front and ego vehicles')
        axes[1, 1].title.set_text('Velocity of front and ego vehicles')
        axes[1, 2].title.set_text('Acceleration of front and ego vehicles')

        axes[1, 0].set_xlabel('Time steps')
        axes[1, 1].set_xlabel('Time steps')
        axes[1, 2].set_xlabel('Time steps')

        axes[0, 0].set_ylabel('Total reward')
        axes[0, 1].set_ylabel('Dist (m)')
        axes[1, 0].set_ylabel('Pos (m)')
        axes[1, 1].set_ylabel('Vel (m/s)')
        axes[1, 2].set_ylabel('Acc')

        axes[1, 0].legend()
        axes[1, 1].legend()
        axes[1, 2].legend()

        fig.tight_layout()
        plt.savefig('img.png')
        plt.show()

    test_agent(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/tmp/sac/sac_for_basic", help="logging directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    test(args)