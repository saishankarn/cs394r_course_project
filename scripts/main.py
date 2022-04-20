# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl 

from stable_baselines3 import A2C

env = gym.make('cruise-ctrl-v0')
s, _ = env.reset()

num_episodes = 1000 

model = A2C("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)