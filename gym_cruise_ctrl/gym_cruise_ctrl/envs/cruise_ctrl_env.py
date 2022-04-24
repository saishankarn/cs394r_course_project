import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from os import path
import pygame
from pygame import gfxdraw
import random

class NoisyDepth():

	def __init__(self):

		self.mean_poly = np.array([2.04935612e-04, -7.82411148e-03, 1.12252551e-01,-6.87136912e-01, 1.62028820e+00, -1.39133046e+00])
		self.std_poly = np.array([-2.07552793e-04, 8.29502928e-03, -1.34784916e-01, 1.03997887e+00, -2.43212328e+00, 2.79613122e+00])
		self.degree = np.shape(self.mean_poly)[0]
		self.bin_range = 5

	def __call__(self, true_depth):
		
		bin_val = int(true_depth / self.bin_range) 
		poly_feat = np.array([bin_val ** i for i in reversed(range(self.degree))])
		mean = np.dot(poly_feat, self.mean_poly) 
		std = np.dot(poly_feat, self.std_poly)

		noise = np.random.normal(mean, std)

		return true_depth + noise


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
		self.violating_safety_dist_reward = -1	# Reward for getting too close to the front car
		self.fv_min_vel = 20 # 20m/s or 50mph
		self.fv_max_vel = 30 # 30m/s or 70mph
		self.delt = 1 # 1s time step 
		self.reward_scaling_const = 1000 # to scale down the reward by this value
		self.sensing_range = 50
		self.depth_noise_model = NoisyDepth()

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
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.InitializeFvVel() 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		# Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32)

		self.state = self.fv_state - self.ego_state # The state is the relative position and speed

		"""
		### Visualizer Parameters
		"""
		self.screen_dim = 500
		self.screen = None

	def InitializeEgoPos(self):
		return 0

	def InitializeEgoVel(self):
		return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph
	
	def InitializeFvPos(self):
		return max(20*np.random.randn() + 100, 10) # (100 +- 20)m

	def InitializeFvVel(self):
		return min(self.fv_max_vel, max(5*np.random.randn() + 25, self.fv_min_vel))	# (25 +-5)m/s or 60mph

	def step(self, action):
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1] 

		# Next state transition
		fv_acc = 0.25*np.random.randn()
		fv_acc = np.clip(fv_acc, self.action_low, self.action_high)
		fv_acc = fv_acc*0#*self.max_acc
		
		action = np.clip(action, self.action_low, self.action_high)[0]
		ego_acc = action#*self.max_acc

		fv_dist_trav = fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		fv_pos = fv_pos + fv_dist_trav 
		fv_vel = fv_vel + fv_acc*self.delt

		ego_dist_trav = ego_vel*self.delt + 0.5*ego_acc*self.delt**2
		ego_pos = ego_pos + ego_dist_trav 
		ego_vel = ego_vel + ego_acc*self.delt

		rel_dis = fv_pos - ego_pos

		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		self.ego_state = np.array([ego_pos, ego_vel], dtype=np.float32)

		self.state = self.fv_state - self.ego_state
		self.state[0] = self.state[0] / self.fv_max_vel
		self.state[1] = self.state[1] / self.sensing_range
		if self.state[1] > 1:
			self.state[1] = 1

		# Reward for the state transition
		reward = (ego_dist_trav - fv_dist_trav) / self.reward_scaling_const
		if rel_dis < self.safety_dist:
			print('collided')
			reward += self.violating_safety_dist_reward 

		# Updating the done variable
		if rel_dis < 0.5 or self.episode_steps >= self.max_episode_steps:
			print("distance remaining : ", rel_dis)
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
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.InitializeFvVel() 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		# Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32) 

		self.init_gap = self.fv_init_pos - self.ego_init_pos

		self.state = self.fv_state - self.ego_state # The state is the relative position and speed
		self.state[0] = self.state[0] / self.fv_max_vel
		self.state[1] = self.state[1] / self.sensing_range
		if self.state[1] > 1:
			self.state[1] = 1

		return self.state

	def render(self, mode="human"):
		
		if self.screen == None:
			pygame.init()
			pygame.display.init()

			self.screen = pygame.display.set_mode([self.screen_dim, self.screen_dim])

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()

		self.screen.fill((255, 255, 255))
		pygame.draw.circle(self.screen, (0,0,255), (self.screen_dim*2/3,self.screen_dim/2), 25)
		pygame.draw.circle(self.screen, (255,0,0), (self.screen_dim*(2/3 - 1/3*self.state[0]/self.fv_init_pos), self.screen_dim/2), 25)
		pygame.display.flip()
		pygame.time.delay(33)

	def close(self):
		pygame.display.quit()
		pygame.quit()