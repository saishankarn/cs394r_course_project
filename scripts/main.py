# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
from cProfile import label
from distutils.log import info
from math import dist
from turtle import color
import gym 
import gym_cruise_ctrl
import matplotlib.pyplot as plt
import numpy as np
from toc import OneDTimeOptimalControl

from stable_baselines3 import PPO, A2C, SAC

env = gym.make('cruise-ctrl-v0')

model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps = 10**5)
# model.save("saved_models/SAC_cruise_ctrl") 
model = SAC.load("saved_models/SAC_cruise_ctrl")

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
