# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
from cProfile import label
from distutils.log import info
from math import dist
from turtle import color
import gym 
import gym_cruise_ctrl
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys 
from toc import OneDTimeOptimalControl

from stable_baselines3 import PPO, A2C, SAC

def run(log_dir):

    """
    ### Initializing the environment, logger, callback and the trainer functions
    """
    env = gym.make('cruise-ctrl-v0') 
    model = SAC("MlpPolicy", env, verbose=1)

    """
    ### Validate results
    """
    model = SAC.load(os.path.join(log_dir, "best_model.zip"))
        
    np.random.seed(0)
    num_eval_episodes = 1000
    random_seeds = np.random.choice(10000, size=(num_eval_episodes,))
    episode_rewards = []

    for ep_idx in range(num_eval_episodes):
        episode_random_seed = random_seeds[ep_idx]

        total_reward_list = [0]

        obs = env.reset(seed=random_seeds[ep_idx], train=False)
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            total_reward_list.append(total_reward_list[-1] + reward)

            if done:
                break

        del total_reward_list[0]
        episode_rewards.append(total_reward_list[-1])

    return 0

if __name__ == "__main__":
    log_dir = sys.argv[1]
    run(log_dir)