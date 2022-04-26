import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
import random
from os import path
import pygame
from pygame import gfxdraw

class NoisyDepth():

	def __init__(self):

		self.mean_poly = np.array([2.04935612e-04, -7.82411148e-03, 1.12252551e-01,-6.87136912e-01, 1.62028820e+00, -1.39133046e+00])
		self.std_poly = np.array([-2.07552793e-04, 8.29502928e-03, -1.34784916e-01, 1.03997887e+00, -2.43212328e+00, 2.79613122e+00])
		self.degree = np.shape(self.mean_poly)[0]
		self.bin_range = 5
		self.min_bin_val = 0 
		self.max_bin_val = 15

	def __call__(self, true_depth):
		
		bin_val = min(max(int(true_depth / self.bin_range) - 1, self.min_bin_val), self.max_bin_val)
		poly_feat = np.array([bin_val ** i for i in reversed(range(self.degree))])
		mean = np.dot(poly_feat, self.mean_poly) 
		std = np.dot(poly_feat, self.std_poly)

		noise = np.random.normal(mean, std)

		return true_depth #+ noise

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
		self.safety_dist = 5	# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward = -10	# Reward for getting too close to the front car
		self.fv_min_vel = 20 # 20m/s or 50mph
		self.fv_max_vel = 30 # 30m/s or 70mph
		self.ego_max_vel = 40 # 40m/s or 90mph
		self.delt = 1 # 1s time step 
		self.depth_noise_model = NoisyDepth()
		self.ego_max_dist = self.ego_max_vel*self.delt # Max distance travelled in one time step
		self.reward_scaling_const = 1000 # to scale down the reward by this value

		"""
		### Episodic Task
		"""
		self.max_episode_steps = 100
		self.episode_steps = 0
		self.done = False

		"""
		### Initial conditions
		"""
		### Front vehicle
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.InitializeFvVel() 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		### Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32)

		self.state = self.fv_state - self.ego_state # The state is the relative position and speed

		"""
		### Visualizer Parameters
		"""
		self.screen_dim = 500
		self.screen = None

		"""
		### Logger Details
		"""
		self.distance_to_return_log = '/tmp/gym/sac/dist_log.txt'
		self.file = open(self.distance_to_return_log, "w+")
		self.training_ep = 0

	def InitializeFvPos(self):
		#return max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		return 100

	def InitializeFvVel(self):
		#return min(self.fv_max_vel, max(5*np.random.randn() + 25, self.fv_min_vel))	# (25 +-5)m/s or 60mph
		return 0

	def InitializeEgoPos(self):
		return 0

	def InitializeEgoVel(self):
		#return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph
		return 0
	



	def step(self, action):
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1] 

		"""
		### Front vehicle state transition
		"""
		### Acceleration input to the front vehicle
		fv_acc = 0.25*np.random.randn()
		fv_acc = np.clip(fv_acc, self.action_low, self.action_high)
		fv_acc = fv_acc*self.max_acc
		fv_acc = 0 # Front vehicle moves with constant velocity
		
		### State update
		fv_dist_trav = fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		fv_pos = fv_pos + fv_dist_trav 
		fv_vel = fv_vel + fv_acc*self.delt
		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		
		"""
		### Ego vehicle state transition
		"""
		### Acceleration input to the ego vehicle
		action = np.clip(action, self.action_low, self.action_high)[0]
		# print(action)
		ego_acc = action*self.max_acc

		### State update
		ego_dist_trav = ego_vel*self.delt + 0.5*ego_acc*self.delt**2
		ego_pos = ego_pos + ego_dist_trav 
		ego_vel = min(ego_vel + ego_acc*self.delt, self.ego_max_vel)
		self.ego_state = np.array([ego_pos, ego_vel], dtype=np.float32)

		"""
		# MDP state update
		"""
		self.state = self.fv_state - self.ego_state
		# self.state[0] = self.state[0] / self.fv_max_vel # Normalizing state. Does this make sense?
		# self.state[1] = self.state[1] / self.init_gap 
		obs = self.state.copy()
		obs[0] = self.depth_noise_model(obs[0])

		
		"""
		# Reward function
		"""
		### Reward for moving forward
		# reward = (ego_dist_trav - fv_dist_trav) / self.ego_max_dist
		reward = ego_dist_trav/self.ego_max_dist
		#reward = reward / 10
		
		### Reward for being too close to the front vehicle
		rel_dis = fv_pos - ego_pos
		if rel_dis < self.safety_dist:
			print('closer than safety distance')
			reward = self.violating_safety_dist_reward

		### Terminating the episode
		if rel_dis < 0.5 or self.episode_steps >= self.max_episode_steps:
			print("distance remaining : ", rel_dis)
			self.file.write(str(rel_dis) + ',' + str(self.training_ep) + "\n")
			self.done = True 

		self.episode_steps += 1

		info = {
			"fv_pos"  : fv_pos,
			"fv_vel"  : fv_vel,
			"ego_pos" : ego_pos,
			"ego_vel" : ego_pos
		}

		return obs, reward, self.done, info

	def reset(self):
		"""
		### Reset the enviornment
		"""
		### Resets the env and returns the starting state and done=False
		### Reset episodic task flags and iterators to initial values
		self.done = False
		self.episode_steps = 0
		self.training_ep += 1
		
		"""
		### Reset states to initial conditions
		"""

		### Front vehicle
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.InitializeFvVel() 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		### Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32) 

		# self.init_gap = self.fv_init_pos - self.ego_init_pos

		### MDP state
		self.state = self.fv_state - self.ego_state # The state is the relative position and speed
		obs = np.zeros((3,))
		obs[0] = self.depth_noise_model(obs[0])
		obs[1] = self.depth_noise_model(obs[0]) 
		


		return obs

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
		pygame.draw.circle(self.screen, (0,0,255), (self.screen_dim*9/10,self.screen_dim/2), 15)
		pygame.draw.circle(self.screen, (255,0,0), (self.screen_dim*(9/10 - 2/10*self.state[0]/self.fv_init_pos), self.screen_dim/2), 15)
		pygame.display.flip()
		pygame.time.delay(33)

	def close(self):
		pygame.display.quit()
		pygame.quit()
