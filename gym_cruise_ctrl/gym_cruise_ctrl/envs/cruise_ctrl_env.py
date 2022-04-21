from math import dist
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2

class CruiseCtrlEnv(gym.Env):

	def __init__(self): 

		"""
		### Action Space
			The action is a scalar in the range `[-1, 1]` that multiplies the max_acc
			to give the acceleration of the ego vehicle. 
		"""
		self.max_acc = 1.5	# 1.5 m/s^2 

		self.action_low = -1.0 
		self.action_high = 1.0
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
		
		"""
		### Observation Space
			The observation is an ndarray of shape (2,) with each element in the range
			`[-inf, inf]`.   
		"""
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

		"""
		### Environment Specifications   
		"""
		self.safety_dist = 2	# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward = -1000	# Reward for getting too close to the front car
		self.fv_min_vel = 20 # 20m/s or 50mph
		self.fv_max_vel = 30 # 30m/s or 70mph
		self.delt = 1 # 1s time step 

		"""
		### Episodic Task
		"""
		self.max_episode_steps = 100
		self.episode_steps = 0
		self.done = False

		"""
		### Initial conditions
		"""
		# Front vehicle
		self.fv_init_pos = max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		self.fv_init_vel = min(self.fv_max_vel, max(5*np.random.randn() + 25, self.fv_min_vel))	# (25 +-5)m/s or 60mph 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		# Ego vehicle
		self.ego_init_pos = 0
		self.ego_init_vel = max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32)

		self.state = self.fv_state - self.ego_state # The state is the relative position and speed

	def step(self, action):
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1]

		# Next state transition
		fv_acc = 0.25*np.random.randn()
		fv_acc = np.clip(fv_acc, self.action_low, self.action_high)[0]
		fv_acc = fv_acc*self.max_acc
		
		action = np.clip(action, self.action_low, self.action_high)[0]
		ego_acc = action*self.max_acc

		fv_pos = fv_pos + fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		fv_vel = fv_vel + fv_acc*self.delt

		dist_trav = ego_vel*self.delt
		ego_pos = ego_pos + dist_trav + 0.5*ego_acc*self.delt**2
		ego_vel = ego_vel + ego_acc*self.delt
		
		rel_dis = fv_pos - ego_pos

		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		self.ego_state = np.array([ego_pos, ego_vel], dtype=np.float32)

		self.state = self.fv_state - self.ego_state

		# Reward for the state transition
		reward = dist_trav
		if rel_dis < self.safety_dist:
			reward += self.violating_safety_dist_reward 

		# Updating the done variable
		if rel_dis < 0.5 or self.episode_steps >= self.max_episode_steps:
			self.done = True 

		self.episode_steps += 1

		return self.state, reward, self.done, {}

	def reset(self):
		# resets the env and returns the starting state and done=False
		# reset episodic task flags and iterators to initial values
		self.done = False
		self.episode_steps = 0
		
		### reset front and ego vehicle states to initial conditions

		# Front vehicle
		self.fv_init_pos = max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		self.fv_init_vel = min(self.fv_max_vel, max(5*np.random.randn() + 25, self.fv_min_vel))	# (25 +-5)m/s or 60mph 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		# Ego vehicle
		self.ego_init_pos = 0
		self.ego_init_vel = max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32)

		self.state = self.fv_state - self.ego_state # The state is the relative position and speed

		return self.state

	def render(self, close=False):
		image = np.zeros((500, 500))
		line1_start_pt = (0, 200)
		line1_end_pt = (500, 200)
		line2_start_pt = (0, 300)
		line2_end_pt = (500, 300)

		cv2.line(image, line1_start_pt, line1_end_pt, 1, 2)
		cv2.line(image, line2_start_pt, line2_end_pt, 1, 2)
		
		
		
		cv2.imshow('cruise control', image)
		cv2.waitKey(5000)
		return None
