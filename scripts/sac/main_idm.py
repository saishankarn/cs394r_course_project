"""
Jai Shri Ram
"""
from cv2 import log
import gym 
import gym_cruise_ctrl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from gym_cruise_ctrl.envs.idm import IDM

from plotting_utils import PlotTestResults

"""
### Script inputs 
"""
env_version = 'cruise-ctrl-v2'
train = False
noisy_depth = False
learning_steps = 10**5

log_dir = 'logs'
load_dir = 'saved_models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(load_dir, exist_ok=True)

"""
### Initializing the environment, logger, callback and the trainer functions
"""

env = gym.make(env_version, train=train, noise_required=noisy_depth)

model = IDM()

"""
### Validate results
"""
obs = env.reset()
ego_vel = env.GetEgoVehicleState()[1]
plot_test_results = PlotTestResults()

while True:
    # action, _ = model.predict(obs)
    action = model.action(obs[0], -obs[1], ego_vel)
    action = np.array([action])
    obs, reward, done, info = env.step(action)
    # env.render()
    
    plot_test_results.store(obs, reward, info) # Gather results for plotting

    if done:
        break
env.close() 

"""
### Generate Plots
"""

plot_test_results.plot()