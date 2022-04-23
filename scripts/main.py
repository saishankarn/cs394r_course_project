# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
from math import dist
import gym 
import gym_cruise_ctrl
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import A2C

env = gym.make('cruise-ctrl-v0')

model = A2C("MlpPolicy", env, verbose=1)

# model.learn(total_timesteps=100000)
# model.save("saved_models/A2C_cruise_ctrl") 

model = A2C.load("saved_models/A2C_cruise_ctrl")

obs = env.reset()
total_reward_list = [0]
dist_list = []
while True:
    
    # action = env.action_space.sample()
    # action = np.array([0.5])
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward_list.append(total_reward_list[-1] + reward)
    dist_list.append(obs[0])
    env.render()
    if done:
        break

env.close() 

fig, axes = plt.subplots(2,1)

axes[0].plot(total_reward_list)
del total_reward_list[0]
axes[1].plot(dist_list)

axes[1].set_xlabel('Time steps')
axes[0].set_ylabel('Total reward accumulated')
axes[1].set_ylabel('Distance between vehicles')

plt.show()
