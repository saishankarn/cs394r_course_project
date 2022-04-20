import gym
from gym import error, spaces, utils
from gym.utils import seeding

class CruiseCtrlEnv(gym.Env):

	def __init__(self):
		
		# environment specifications
		self.max_acc = 0.5 
		self.safety_dist = 2
		self.violating_safety_dist_reward = -10
		self.allowed_actions = [-1, -0.5, 0, 0.5, 1]

		self.max_episode_steps = 100
		self.episode_steps = 0
		self.done = False

		# front vehicle specifications and variables
		self.fv_start_dist = 50 
		self.fv_dist = 0

		# ego vehicle variables
		self.ego_vel = 0
		self.ego_acc = 0

	def step(self, action):
		# next state transition

		acc = self.allowed_actions[action] * self.max_acc  
		dist_traveled = self.ego_vel + 0.5*acc
		self.ego_vel += acc
		self.fv_dist -= dist_traveled

		next_state = [self.ego_vel, self.fv_dist]

		# reward for the state transition
		reward = dist_traveled
		if self.fv_dist < self.safety_dist:
			reward += self.violating_safety_dist_reward 

		# updating the done variable
		if self.fv_dist <= 0.5 or self.episode_steps >= self.max_episode_steps:
			self.done = True 

		self.episode_steps += 1

		return next_state, reward, self.done

	def reset(self):
		# resets the env and returns the starting state and done=False
		self.fv_dist = self.fv_start_dist
		self.ego_vel = 0 
		self.ego_acc = 0
		self.done = False
		self.episode_steps = 0
		state = [self.ego_vel, self.fv_dist]

		return state, self.done

	def render(self, close=False):
		return None
