import numpy as np
from distutils.log import info
import gym 
import gym_cruise_ctrl
import matplotlib.pyplot as plt
import numpy as np
from mpc import MPCLinear

from stable_baselines3 import SAC

# The code tries to implement an RL agent to the cruise-ctrl-v0 env 

env = gym.make('cruise-ctrl-v0', train=True, noise_required=False) 

model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps = 10**5)
# model.save("saved_models/SAC_cruise_ctrl") 
model = SAC.load("saved_models/SAC_cruise_ctrl")

"""
### Initialize model predictive controller
"""
Ts = 1
A = np.array([[1, Ts],
              [0,  1]])
B = 1.5*np.array([[-0.5*Ts*Ts],
              [-Ts]])
T = 10
Q = np.diag([5, 5])
R = np.array([5])
modelMPC = MPCLinear(A, B, Q, R, T)
# x0 = [10, 2]
xref = [5, 0]

"""
### Validate results
"""
action_list = []
total_reward_list = [0]
rel_dist_list = []
fv_pos_list = []
ego_pos_list = []
fv_acc_list = []
ego_acc_list = []

obs = env.reset()

while True:
    # action, _ = model.predict(obs)
    
    u, _ = modelMPC.action(obs, xref)
    if u is None:
        action = np.array([0])
    else:
        action = np.array([u[0,0]])
    obs, reward, done, info = env.step(action)
    # env.render()

    # Gather results for plotting
    action_list.append(action)
    total_reward_list.append(total_reward_list[-1] + reward)
    rel_dist_list.append(obs[0])
    fv_pos_list.append(info["fv_pos"])
    ego_pos_list.append(info["ego_pos"])
    fv_acc_list.append(info["fv_acc"])
    ego_acc_list.append(info["ego_acc"])

    
    if done:
        break

del total_reward_list[0]
# env.close() 

"""
### Generate Plots
"""
# print(total_reward_list)
# print(rel_dist_list)
# print(fv_pos_list)
# print(ego_pos_list)
fig, axes = plt.subplots(2,2, figsize = (10,10))

axes[0,0].plot(total_reward_list)
axes[1,0].plot(rel_dist_list)
axes[0,1].plot(fv_pos_list, color = 'b', label = 'Front vehicle')
axes[0,1].plot(ego_pos_list, color = 'r',  label = 'Ego vehicle')
axes[1,1].plot(fv_acc_list, color = 'b', label = 'Front vehicle')
axes[1,1].plot(ego_acc_list, color = 'r',  label = 'Ego vehicle')

axes[0,1].legend()
axes[1,1].legend()
axes[0,0].title.set_text('Total reward accumulated over time')
axes[1,0].title.set_text('Relative distance between vehicles over time')
axes[0,1].title.set_text('Positions of front and ego vehicles')
axes[1,1].title.set_text('Accelerations of front and ego vehicles')

axes[0,0].set_xlabel('Time steps')
axes[1,1].set_xlabel('Time steps')

axes[0,0].set_ylabel('Total reward')
axes[1,0].set_ylabel('Distance (m)')
axes[0,1].set_ylabel('Position (m)')
axes[1,1].set_ylabel('Acceleration (m/s)')

axes[0,0].set_xlim([0, 100])
axes[1,0].set_xlim([0, 100])
axes[0,1].set_xlim([0, 100])
axes[1,1].set_xlim([0, 100])

fig.tight_layout()
plt.show()
