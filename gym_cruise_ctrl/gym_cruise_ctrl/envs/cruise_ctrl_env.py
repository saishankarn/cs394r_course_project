import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class CruiseCtrlEnv(gym.Env):

	def __init__(self):
		
		# environment specifications
		self.max_acc = 0.5 
		self.safety_dist = 2
		self.violating_safety_dist_reward = -10

		self.max_episode_steps = 100
		self.episode_steps = 0
		self.done = False

		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
		self.action_low = -1.0 
		self.action_high = 1.0

		# front vehicle specifications and variables
		self.fv_start_dist = 50 
		self.fv_dist = 0

		# ego vehicle variables
		self.ego_vel = 0
		self.ego_acc = 0

	def step(self, action):
		# next state transition
		action = np.clip(action, self.action_low, self.action_high)[0]
		acc = action * self.max_acc  
		dist_traveled = self.ego_vel + 0.5*acc
		self.ego_vel += acc
		self.fv_dist -= dist_traveled

		next_state = np.array([self.ego_vel, self.fv_dist], dtype=np.float32)

		# reward for the state transition
		reward = dist_traveled
		if self.fv_dist < self.safety_dist:
			reward += self.violating_safety_dist_reward 

		# updating the done variable
		if self.fv_dist <= 0.5 or self.episode_steps >= self.max_episode_steps:
			self.done = True 

		self.episode_steps += 1

		return next_state, reward, self.done, {}

	def reset(self):
		# resets the env and returns the starting state and done=False
		self.fv_dist = self.fv_start_dist
		self.ego_vel = 0 
		self.ego_acc = 0
		self.done = False
		self.episode_steps = 0
		state = np.array([self.ego_vel, self.fv_dist], dtype=np.float32)

		return state

	def render(self, close=False):
		return None
