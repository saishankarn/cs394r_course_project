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


def run():

    """
    ### Initializing the environment, logger, callback and the trainer functions
    """
    log_dir = "/tmp/gym/sac_with_noise_in_both_states"
    #log_dir = "/tmp/gym/sac"
    os.makedirs(log_dir, exist_ok = True)

    env = gym.make('cruise-ctrl-v0') 
    env = Monitor(env, log_dir) # Logs will be saved in log_dir/monitor.csv 

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq = 1000, log_dir = log_dir)

    model = SAC("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps = 10**6, callback = callback)
    #model.save("saved_models/SAC_cruise_ctrl_with_depth_uncertainty_and_randomness") 

    """
    ### Validate results
    """
    #model = SAC.load("saved_models/SAC_cruise_ctrl")
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

    fig, axes = plt.subplots(5,1)

    axes[0].plot(total_reward_list)
    axes[1].plot(rel_dist_list)
    axes[2].plot(fv_pos_list, color = 'b', label = 'Front vehicle')
    axes[2].plot(ego_pos_list, color = 'r',  label = 'Ego vehicle')
    axes[3].plot(fv_vel_list, color = 'b', label = 'Front vehicle')
    axes[3].plot(ego_vel_list, color = 'r',  label = 'Ego vehicle')
    axes[4].plot(fv_acc_list, color = 'b', label = 'Front vehicle')
    axes[4].plot(ego_acc_list, color = 'r',  label = 'Ego vehicle')

    axes[0].title.set_text('Total reward accumulated over time')
    axes[1].title.set_text('Distance between vehicles over time')
    axes[2].title.set_text('Position of front and ego vehicles')
    axes[3].title.set_text('Velocity of front and ego vehicles')
    axes[4].title.set_text('Acceleration of front and ego vehicles')

    axes[2].set_xlabel('Time steps')
    axes[3].set_xlabel('Time steps')
    axes[4].set_xlabel('Time steps')

    axes[0].set_ylabel('Total reward')
    axes[1].set_ylabel('Dist (m)')
    axes[2].set_ylabel('Pos (m)')
    axes[3].set_ylabel('Vel (m/s)')
    axes[4].set_ylabel('Acc')

    plt.legend
    plt.savefig('img.png')

    return 0

if __name__ == "__main__":
    run()


"""
env = gym.make('cruise-ctrl-v0')

model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps = 10**5)
# model.save("saved_models/SAC_cruise_ctrl") 
model = SAC.load("saved_models/SAC_cruise_ctrl")

model = SAC.load("saved_models/SAC_cruise_ctrl")

total_reward_list = [0]
rel_dist_list = []
fv_pos_list = []
ego_pos_list = []

obs = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    #env.render()

    # Gather results for plotting
    total_reward_list.append(total_reward_list[-1] + reward)
    rel_dist_list.append(obs[0])
    fv_pos_list.append(info["fv_pos"])
    ego_pos_list.append(info["ego_pos"])

    
    if done:
        break

del total_reward_list[0]
env.close() 

"""
### Generate Plots
"""
# print(total_reward_list)
# print(rel_dist_list)
# print(fv_pos_list)
# print(ego_pos_list)
fig, axes = plt.subplots(3,1)

axes[0].plot(total_reward_list)
axes[1].plot(rel_dist_list)
axes[2].plot(fv_pos_list, color = 'b', label = 'Front vehicle')
axes[2].plot(ego_pos_list, color = 'r',  label = 'Ego vehicle')

axes[0].title.set_text('Total reward accumulated over time')
axes[1].title.set_text('Distance between vehicles over time')
axes[2].title.set_text('Position of front and ego vehicles')

axes[2].set_xlabel('Time steps')

axes[0].set_ylabel('Total reward')
axes[1].set_ylabel('Distance (m)')
axes[2].set_ylabel('Position (m)')

plt.legend
#plt.show()
plt.savefig('img.png')




"""
### Initialize time optimal controller
"""
toc = OneDTimeOptimalControl(*env.GetTOCInitParams())

"""
### Validate results
"""
total_reward_list = [0]
rel_dist_list = []
fv_pos_list = []
ego_pos_list = []
action_list = []

obs = env.reset()

while True:
    # action, _ = model.predict(obs)
    ego_state = env.GetEgoVehicleState()
    action = np.array([toc.action(ego_state[1], obs[0])])
    # action = np.array([toc.action(obs[1], obs[0])])
    obs, reward, done, info = env.step(action)
    env.render()

    # Gather results for plotting
    action_list.append(action)
    total_reward_list.append(total_reward_list[-1] + reward)
    rel_dist_list.append(obs[0])
    fv_pos_list.append(info["fv_pos"])
    ego_pos_list.append(info["ego_pos"])

    
    if done:
        break

del total_reward_list[0]
env.close() 

"""
### Generate Plots
"""
# print(total_reward_list)
# print(rel_dist_list)
# print(fv_pos_list)
# print(ego_pos_list)
fig, axes = plt.subplots(2,2)

axes[0,0].plot(total_reward_list)
axes[1,0].plot(rel_dist_list)
axes[0,1].plot(fv_pos_list, color = 'b', label = 'Front vehicle')
axes[0,1].plot(ego_pos_list, color = 'r',  label = 'Ego vehicle')
axes[1,1].plot(action_list)

axes[0,0].title.set_text('Total reward accumulated over time')
axes[1,0].title.set_text('Relative distance between vehicles over time')
axes[0,1].title.set_text('Position of front and ego vehicles')
axes[1,1].title.set_text('Ego vehicle throttle')

axes[0,0].set_xlabel('Time steps')
axes[1,1].set_xlabel('Time steps')

axes[0,0].set_ylabel('Total reward')
axes[1,0].set_ylabel('Distance (m)')
axes[0,1].set_ylabel('Position (m)')
axes[1,1].set_ylabel('Throttle')

plt.legend
plt.show()
"""