"""
Jai Shri Ram

CruiseCtrlEnv4 has four observation variables
1. Relative distance
2. Relative velocity
3. IDM action
4. Previous acceleration / action

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
from gym_cruise_ctrl.envs.idm import IDM 

class CruiseCtrlEnv4(gym.Env):

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
			The observation is an ndarray of shape (3,) with each element in the range
			`[-inf, inf]`.   
			1. Relative distance
			2. Relative velocity
			3. Previous action
		"""
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))


		"""
		### Episodic Task
		"""
		self.delt = 1 # 0.1s time step
		self.max_episode_steps = int(100/self.delt)
		print(self.max_episode_steps)
		self.episode_steps = 0
		self.done = False

		"""
		### Environment Specifications   
		"""
		self.safety_dist = 5	# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward = -10*self.delt	# Reward for getting too close to the front car
		self.fv_min_vel = 5 # 10m/s or 30mph
		self.fv_max_vel = 15 # 30m/s or 70mph
		self.fv_max_acc = 0.5  # 0.5m/s^2 
		self.ego_max_vel = 40 # 40m/s or 90mph 
		self.ego_max_dist = self.ego_max_vel#*self.delt # Max distance travelled in one time step


		self.fv_vel_list, self.fv_acc_list = Spline(self.max_episode_steps, 4, 8)
		self.fv_vel_list = self.fv_min_vel + self.fv_vel_list*(self.fv_max_vel - self.fv_min_vel)
		self.fv_acc_list = self.fv_acc_list*(self.fv_max_vel - self.fv_min_vel)/self.delt 

		self.classic_control = IDM() # classic control model 
		self.depth_noise_model = NoisyDepth() # depth noise model class
		self.vel_noise_model = NoisyVel() # velocity noise model class 
		self.noise_required = noise_required # whether noise is required or not
		self.jerk_cost = 1

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

		"""
		### The state doesnot have IDM action as the IDM action can correspond to noisy depth and relative velocity
		### The state has the following three elements
		1. Relative position (depth)
		2. Relative velocity
		3. Previously executed acceleration command
		"""
		
		rel_pose = self.fv_state - self.ego_state # The state is the relative position and speed
		self.state = np.append(rel_pose, self.ego_init_acc)

		"""
		### Visualizer Parameters
		"""
		self.screen_dim = 500
		self.screen = None


	def InitializeFvPos(self):
		if self.train:
			return 100
			#return max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		else:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m

	def InitializeFvVel(self):
		if self.train:
			return 0
		else:
			#return min(self.fv_max_vel, max(5*np.random.randn() + 20, self.fv_min_vel))	# (25 +-5)m/s or 60mph
			return 10

	def InitializeEgoPos(self):
		return 0

	def InitializeEgoVel(self):
		if self.train:
			return 0
			#return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s or 30mph
		else:
			return self.fv_init_vel# + np.random.uniform(-2, +2)	# (10 +-5)m/s or 30mph
			#return 10

	def step(self, action):

		# current state of the front vehicle and the ego vehicle 
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1] 

		"""
		### Front vehicle state transition
		"""	

		### State update
		fv_acc = self.fv_acc_list[self.episode_steps]
		fv_vel = self.fv_vel_list[self.episode_steps]
		if self.train:
			fv_acc = 0 
			fv_vel = 0
		fv_pos = fv_pos + fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		
		"""
		### Ego vehicle state transition
		"""
		### Acceleration input to the ego vehicle
		action = np.clip(action, -self.max_acc, self.max_acc)[0]
		ego_acc = action.item() 
		
		# Clamping the action by activating the shield
		'''
		ego_acc = action[0]
		
		activate_shield = False 
		if (ego_acc < self.state[-1] - 0.05) or (ego_acc < self.state[-1] + 0.05):
			activate_shield = True 

		ego_acc = np.clip(ego_acc, self.state[-1] - 0.05, self.state[-1] + 0.05)

		# clamping the action to lie within bounds
		ego_acc = np.clip(ego_acc, -self.max_acc, self.max_acc)
		'''	
		### Clipping acceleration to keep within velocity limits
		if ego_vel >= self.ego_max_vel:
			if ego_vel + ego_acc*self.delt >= self.ego_max_vel:
				ego_acc = 0.0
		else:
			if ego_vel + ego_acc*self.delt >= self.ego_max_vel:
				ego_acc = (self.ego_max_vel - ego_vel)/self.delt 


		### Environment State update
		ego_dist_trav = ego_vel*self.delt + 0.5*ego_acc*self.delt**2
		ego_pos = ego_pos + ego_dist_trav 
		ego_vel = ego_vel + ego_acc*self.delt
		self.ego_state = np.array([ego_pos, ego_vel], dtype=np.float32)

		self.ego_jerk = (ego_acc - self.state[-1])/self.delt

		rel_pose = self.fv_state - self.ego_state

		"""
		# State update
		"""
		self.state = rel_pose.copy()
		self.state = np.append(self.state, ego_acc)

		"""
		# Noise corruption
		"""
		if self.noise_required:
			rel_pose[1] = self.vel_noise_model(rel_pose[1], rel_pose[0])
			rel_pose[0] = self.depth_noise_model(rel_pose[0])

		"""
		# Observation update
		"""
		control_command = self.classic_control.action(rel_pose[0], -rel_pose[1], self.ego_state[1])
		obs = np.append(rel_pose.copy(), control_command)
		obs = np.append(obs, self.state[-1])

		"""
		# Reward function
		"""
		### Reward for moving forward
		reward = ego_dist_trav/self.ego_max_dist

		### Penalty for high jerk
		#reward -= abs(self.ego_jerk) * self.jerk_cost 

		### Penalty for being too close to the front vehicle
		rel_dis = fv_pos - ego_pos
		if rel_dis < self.safety_dist:
			reward += self.violating_safety_dist_reward

		"""
		# Checking for the termination condition
		"""
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

		"""
		# State update
		"""
		self.state = rel_pose.copy()
		self.state = np.append(self.state, self.ego_init_acc)


		"""
		# Noise corruption
		"""
		if self.noise_required:
			rel_pose[1] = self.vel_noise_model(rel_pose[1], rel_pose[0])
			rel_pose[0] = self.depth_noise_model(rel_pose[0])

		"""
		# Observation update
		"""
		control_command = self.classic_control.action(rel_pose[0], -rel_pose[1], self.ego_state[1])
		obs = np.append(rel_pose.copy(), control_command)
		obs = np.append(obs, self.state[-1])

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