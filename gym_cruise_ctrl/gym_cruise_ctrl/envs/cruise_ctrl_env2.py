"""
Jai Shri Ram 
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from os import path
import pygame
from pygame import gfxdraw
from gym_cruise_ctrl.envs.input_generator import PiecewiseLinearProfile, Spline
from gym_cruise_ctrl.envs.noise_model import NoisyDepth, NoisyVel

class CruiseCtrlEnv2(gym.Env):

	def __init__(self, train=True, noise_required=False): 

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
			1. Relative distance
			2. Relative velocity
			3. Previous action
		"""
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))


		"""
		### Episodic Task
		"""
		self.delt = 0.1 # 0.1s time step
		self.max_episode_steps = int(100/self.delt)
		self.episode_steps = 0
		self.done = False

		"""
		### Environment Specifications   
		"""
		self.safety_dist = 5	# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward = -10*self.delt	# Reward for getting too close to the front car
		self.fv_min_vel = 10 # 10m/s or 30mph
		self.fv_max_vel = 30 # 30m/s or 70mph
		self.fv_max_acc = 0.5  # 0.5m/s^2
		self.ego_max_vel = 40 # 40m/s or 90mph 
		self.ego_max_dist = self.ego_max_vel*self.delt # Max distance travelled in one time step
		# self.fv_input_gen = PiecewiseLinearProfile(self.max_episode_steps, 1, 5) # class to generate the behaviour of the front vehicle
		# self.fv_input = self.fv_input_gen.generate() # the behaviour of the front vehicle
		self.fv_vel_list, self.fv_acc_list = Spline(self.max_episode_steps, 4, 8)
		self.fv_vel_list = self.fv_min_vel + self.fv_vel_list*(self.fv_max_vel - self.fv_min_vel)
		self.fv_acc_list = self.fv_acc_list*(self.fv_max_vel - self.fv_min_vel)/self.delt
		
		self.depth_noise_model = NoisyDepth() # depth noise model class
		self.vel_noise_model = NoisyVel() # velocity noise model class 
		self.noise_required = noise_required # whether noise is required or not
		self.jerk_cost = 10
		self.nominal_max_gap_keeping_error = 15

		### for seed purposes 
		self.train = train # are we training or validating? For validating, we set the seed to get constant initializations

		"""
		### Initial conditions
		"""
		### Front vehicle
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.fv_vel_list[0]
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)

		### Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32) 

		self.ego_init_acc = 0.0
		self.ego_jerk	  = 0.0
		
		rel_pose = self.fv_state - self.ego_state # The state is the relative position and speed
		self.state = np.append(rel_pose, self.ego_init_acc)

		"""
		### Visualizer Parameters
		"""
		self.screen_dim = 500
		self.screen = None


	def InitializeFvPos(self):
		if self.train:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		else:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m

	def InitializeFvVel(self):
		if self.train:
			return 0
		else:
			return min(self.fv_max_vel, max(5*np.random.randn() + 20, self.fv_min_vel))	# (25 +-5)m/s or 60mph

	def InitializeEgoPos(self):
		return 0

	def InitializeEgoVel(self):
		if self.train:
			return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph
		else:
			return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph


	def step(self, action):
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1] 

		"""
		### Front vehicle state transition
		"""
		### Acceleration input to the front vehicle
		# fv_acc = self.fv_input[self.episode_steps]
		# fv_acc = np.clip(fv_acc, self.action_low, self.action_high)
		# if self.train:
		# 	fv_acc = 0.0
		# else:
		# 	fv_acc = fv_acc*self.fv_max_acc*0
		
		# ### Clipping acceleration to keep within velocity limits
		# if fv_vel >= self.fv_max_vel:
		# 	if fv_vel + fv_acc*self.delt >= self.fv_max_vel:
		# 		fv_acc = 0.0
		# else:
		# 	if fv_vel + fv_acc*self.delt >= self.fv_max_vel:
		# 		fv_acc = (self.fv_max_vel - fv_vel)/self.delt
		
		# if fv_vel <= self.fv_min_vel:
		# 	if fv_vel + fv_acc*self.delt <= self.fv_min_vel:
		# 		fv_acc = 0.0
		# else:
		# 	if fv_vel + fv_acc*self.delt <= self.fv_min_vel:
		# 		fv_acc = (self.fv_min_vel - fv_vel)/self.delt		

		### State update
		fv_acc = self.fv_acc_list[self.episode_steps]
		fv_vel = self.fv_vel_list[self.episode_steps]
		fv_pos = fv_pos + fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		
		"""
		### Ego vehicle state transition
		"""
		### Acceleration input to the ego vehicle
		action = np.clip(action, -self.max_acc, self.max_acc)[0]
		ego_acc = action.item() #*self.max_acc

		### Clipping acceleration to keep within velocity limits
		if ego_vel >= self.ego_max_vel:
			if ego_vel + ego_acc*self.delt >= self.ego_max_vel:
				ego_acc = 0.0
		else:
			if ego_vel + ego_acc*self.delt >= self.ego_max_vel:
				ego_acc = (self.ego_max_vel - ego_vel)/self.delt

		### State update
		ego_dist_trav = ego_vel*self.delt + 0.5*ego_acc*self.delt**2
		ego_pos = ego_pos + ego_dist_trav 
		ego_vel = ego_vel + ego_acc*self.delt
		self.ego_state = np.array([ego_pos, ego_vel], dtype=np.float32)

		self.ego_jerk = (ego_acc - self.state[2])/self.delt

		"""
		# Noise corruption
		"""
		rel_pose = self.fv_state - self.ego_state
		self.state = np.append(rel_pose.copy(), ego_acc)

		if self.noise_required:
			rel_pose[1] = self.vel_noise_model(rel_pose[1], rel_pose[0])
			rel_pose[0] = self.depth_noise_model(rel_pose[0])

		"""
		# Observation update
		"""
		obs = np.append(rel_pose.copy(), ego_acc)
		
		"""
		# Reward function
		"""
		### Reward for moving forward
		reward = ego_dist_trav/self.ego_max_dist
		
		### ACC cost function
		# cost_gap_keeping_error = abs((rel_pose[0] - self.safety_dist)/self.nominal_max_gap_keeping_error)
		cost_control_effort    = abs(action/max(abs(self.action_low), abs(self.action_high)))
		cost_jerk			   = abs(self.ego_jerk/(self.max_acc*(self.action_high - self.action_low)/self.delt))
		cost_jerk              = abs(self.ego_jerk)

		reward = reward - 0*cost_control_effort - 0.0*cost_jerk

		### Reward for being too close to the front vehicle
		rel_dis = fv_pos - ego_pos
		if rel_dis < self.safety_dist:
			reward += self.violating_safety_dist_reward

		### Terminating the episode
		if rel_dis <= 2 or self.episode_steps >= self.max_episode_steps:
			print("distance remaining : ", rel_dis)
			self.done = True 

		self.episode_steps += 1

		info = {
			"fv_pos"  : fv_pos,
			"fv_vel"  : fv_vel,
			"fv_acc"  : fv_acc, 
			"ego_pos" : ego_pos,
			"ego_vel" : ego_vel,
			"ego_acc" : ego_acc
		}

		return obs, reward, self.done, info

	def reset(self, seed=0):
		"""
		### setting the fixed seed for validation purposes
		"""
		# if not self.train:
		# 	np.random.seed(seed)
		
		"""
		### Reset the enviornment
		"""
		### Resets the env and returns the starting state and done=False
		### Reset episodic task flags and iterators to initial values
		self.done = False
		self.episode_steps = 0
		
		"""
		### Reset states to initial conditions
		"""

		### Front vehicle
		self.fv_vel_list, self.fv_acc_list = Spline(self.max_episode_steps, 4, 8)
		self.fv_vel_list = self.fv_min_vel + self.fv_vel_list*(self.fv_max_vel - self.fv_min_vel)
		self.fv_acc_list = self.fv_acc_list*(self.fv_max_vel - self.fv_min_vel)/self.delt

		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.fv_vel_list[0] 
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)
		# self.fv_input = self.fv_input_gen.generate()

		### Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32)

		self.ego_init_acc = 0.0
		self.ego_jerk     = 0.0
		
		rel_pose = self.fv_state - self.ego_state # The state is the relative position and speed
		self.state = np.append(rel_pose, self.ego_init_acc)

		### Observation 
		obs = self.state.copy()

		return obs 

	def render(self, mode='human'):
		
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

	def GetTOCInitParams(self):
		return self.ego_max_vel, self.max_acc, self.max_acc, self.delt

	def GetEgoVehicleState(self):
		return self.ego_state

