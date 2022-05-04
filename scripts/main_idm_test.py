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

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from saving_utils import SaveOnBestTrainingRewardCallback
from plotting_utils import PlotTrainResults, PlotTestResults

"""
### Script inputs 
"""
env_version = 'cruise-ctrl-v2'
train = False
noisy_depth = False
model_name = 'sb_SAC_acc_rl_jl_wo_noise'
learning_steps = 10**5

log_dir = 'logs'
load_dir = 'saved_models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(load_dir, exist_ok=True)

"""
### Initializing the environment, logger, callback and the trainer functions
"""

env = gym.make(env_version, train=train, noise_required=noisy_depth)
# model = SAC("MlpPolicy", env, verbose=0)
model = IDM()

# if train == True:
#     env = Monitor(env, log_dir) # Logs will be saved in log_dir/monitor.csv 
#     callback = SaveOnBestTrainingRewardCallback(check_freq = 1000, 
#                                                 log_dir    = log_dir,
#                                                 save_dir   = load_dir,
#                                                 model_name = model_name) # Create the callback: check every 1000 steps
#     model = SAC("MlpPolicy", env, verbose=0)
#     model.learn(total_timesteps = learning_steps, callback = callback)
#     PlotTrainResults(log_dir)

"""
### Validate results
"""
# model = SAC.load(os.path.join(load_dir, model_name))
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