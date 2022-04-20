# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl 

from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN

env = gym.make('cruise-ctrl-v0')
s, _ = env.reset()

num_episodes = 1000 

state_size = len(s)
action_size = len(env.allowed_actions)

lr = 0.001 

pi = PiApproximationWithNN(state_size, action_size, lr)
B = VApproximationWithNN(state_size, lr)

REINFORCE(env,1,1000,pi,B)
