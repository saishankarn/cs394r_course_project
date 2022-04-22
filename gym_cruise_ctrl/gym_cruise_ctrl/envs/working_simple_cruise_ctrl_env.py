import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np 

import cv2 

class CruiseCtrlEnv(gym.Env):

	def __init__(self):
		
		# environment specifications
		"""
		### Action Space
			The action is a scalar in the range `[-1, 1]` that multiplies the max_acc
			to give the acceleration of the ego vehicle. 
		"""
		self.max_acc = 0.1 
		self.action_low = -1.0 
		self.action_high = 1.0
		self.action_space = gym.spaces.Box(low=self.action_low, high=self.action_high, shape=(1,))

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
		self.violating_safety_dist_reward = -1	# Reward for getting too close to the front car

		self.fv_min_vel = 20 # 20m/s or 50mph
		self.fv_max_vel = 30 # 30m/s or 70mph
		self.delt = 1 # 1s time step 

		self.max_episode_steps = 100
		self.episode_steps = 0
		self.done = False		

		# front vehicle specifications and variables
		self.fv_start_dist = 50 
		self.fv_dist = 0

		# ego vehicle variables
		self.ego_vel = 0
		self.max_vel_possible = 10

	def step(self, action):
		# next state transition
		#print("within step")
		action = np.clip(action, self.action_low, self.action_high)[0]
		acc = action * self.max_acc
		dist_traveled = self.ego_vel*self.delt + 0.5*acc
		#print("distance traveled : ", dist_traveled)
		self.ego_vel += acc
		#print("checking fv dist value : ", self.fv_dist, self.fv_min_vel, dist_traveled)
		self.fv_dist -= dist_traveled
		#print("code value : ", self.fv_dist)

		next_state = np.array([self.ego_vel/self.max_vel_possible, self.fv_dist/self.fv_start_dist], dtype=np.float32)

		# reward for the state transition
		reward = dist_traveled / 100
		#print(next_state, reward)
		if self.fv_dist < self.safety_dist:
			print('collided')
			reward += self.violating_safety_dist_reward 

		# updating the done variable
		if self.fv_dist <= 0.5 or self.episode_steps >= self.max_episode_steps:
			print("distance remaining : ", self.fv_dist)
			self.done = True 

		self.episode_steps += 1
		#print("episode details : ", self.episode_steps, self.done)

		return next_state, reward, self.done, {}

	def reset(self):
		# resets the env and returns the starting state and done=False
		self.fv_dist = self.fv_start_dist 
		self.ego_vel = 0
		self.done = False
		self.episode_steps = 0
		state = np.array([self.ego_vel/self.max_vel_possible, self.fv_dist/self.fv_start_dist], dtype=np.float32)
		#print("reset state : ", state)
		return state

	def render(self, close=False):
		image = np.zeros((500, 500))
		line1_start_pt = (0, 250) 
		line1_end_pt = (500, 250)
		line2_start_pt = (0, 350)
		line2_end_pt = (500, 350)

		cv2.line(image, line1_start_pt, line1_end_pt, 1, 2)
		cv2.line(image, line2_start_pt, line2_end_pt, 1, 2)
		cv2.imshow('cruise control', image)
		cv2.waitKey(1000)
		return None