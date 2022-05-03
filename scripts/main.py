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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True


def run(log_dir, train, noise_required):

    """
    ### Initializing the environment, logger, callback and the trainer functions
    """
    print("noise required : ", noise_required)
    env = gym.make('cruise-ctrl-v0', train=train, noise_required=noise_required) 
    env = Monitor(env, log_dir) # Logs will be saved in log_dir/monitor.csv 

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq = 1000, log_dir = log_dir)

    model = SAC("MlpPolicy", env, verbose=1)

    if train:
        model.learn(total_timesteps = 10**6, callback = callback)


    """
    ### Validate results
    """
    model = SAC.load(os.path.join(log_dir, "best_model.zip"))

    total_reward_list = [0]
    rel_dist_list = []
    fv_pos_list = []
    fv_vel_list = []
    fv_acc_list = []
    ego_pos_list = []
    ego_vel_list = []
    ego_acc_list = []

    obs = env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()

        # Gather results for plotting
        total_reward_list.append(total_reward_list[-1] + reward)
        rel_dist_list.append(obs[0])

        fv_pos_list.append(info["fv_pos"])
        fv_vel_list.append(info["fv_vel"])
        fv_acc_list.append(info["fv_acc"])

        ego_pos_list.append(info["ego_pos"])
        ego_vel_list.append(info["ego_vel"])
        ego_acc_list.append(info["ego_acc"])


        if done:
            break

    del total_reward_list[0]
    env.close() 

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

    return 0

if __name__ == "__main__":
    log_dir = sys.argv[1] # path (str) to log the results of the experiment 
    train = sys.argv[2] # bool (True or False) whether to train or not 
    noise_required = sys.argv[3] # bool (True or False) whether noise from depth and velocity is required or not
    os.makedirs(log_dir, exist_ok=True)

    train = True 
    noise_required = False
    run(log_dir, train, noise_required)