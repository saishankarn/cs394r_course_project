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
from gym_cruise_ctrl.envs.input_generator2 import InputAccGenerator
from gym_cruise_ctrl.envs.idm import IDM
from gym_cruise_ctrl.envs.network_strength import NetworkStrength
from gym_cruise_ctrl.envs.delay_handler import DelayHandler

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

		return true_depth + noise

class NoisyVel():

	def __init__(self):

		self.mean = 0
		self.std = 1
		self.min_vel = 20 
		self.max_vel = 30
		self.bin_range = 5
		self.min_bin_val = 0 
		self.max_bin_val = 19
		self.num_bins = 20


	def __call__(self, true_vel, true_depth):

		bin_val = min(max(int(true_depth / self.bin_range) - 1, self.min_bin_val), self.max_bin_val)
		std_bin = (self.std / self.num_bins) * bin_val
		noise = np.random.normal(self.mean, std_bin)
		
		return true_vel + noise

class CruiseCtrlEnv2(gym.Env):

	def __init__(self, train=True, noise_required=False): 

		"""
		### Action Space
			The action is a scalar in the range `[-1, 1]` that multiplies the max_acc
			to give the acceleration of the ego vehicle. 
		"""
		self.max_acc 	  = 1.5	# 1.5 m/s^2 

		self.action_low   = -1.0
		self.action_high  =  1.0
		self.action_space = gym.spaces.Box(low=self.action_low, high=self.action_high, shape=(1,))
		
		"""
		### Observation Space
			The observation is an ndarray of shape (8,) with each element in the range
			`[-inf, inf]`.   
			1. Relative distance-1 (noisy)
			2. Relative velocity-1 (noisy)
			3. IDM action-1 (noisy)
			4. Relative distance-2 (laggy)
			5. Relative velocity-2 (laggy)
			6. IDM action-2 (laggy)
			7. Ego vehicle acceleration
			8. Network strength
		"""
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))

		"""
		### Episodic Task
		"""
		self.delt = 1										# 0.1s time step
		self.max_episode_steps = int(100/self.delt)			# 1000 episode length

		"""
		### Environment Specifications   
		"""
		### Safety specifications
		self.safety_dist 				   = 5				# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward  = -10*self.delt	# Reward for getting too close to the front car
		self.jerk_cost_coef				   = 0.1

		### Front vehicle specifications
		self.fv_min_vel = 10 								# 10m/s or 30mph
		self.fv_max_vel = 30 								# 30m/s or 70mph
		self.fv_max_acc = 0.5  								# 0.5m/s^2

		### Ego vehicle specifications		
		self.ego_max_vel  = 40 								# 40m/s or 90mph 
		self.ego_max_dist = self.ego_max_vel*self.delt      # Max distance travelled in one time step
		
		### Noise specifications
		self.noise_required    = noise_required 			# Whether noise is required or not
		self.depth_noise_model = NoisyDepth() 				# Depth noise model class
		self.vel_noise_model   = NoisyVel() 				# Velocity noise model class 

		### For random seed purposes 
		self.train = train 									# Are we training or validating? For validating, we set the seed to get constant initializations

		"""
		### Visualizer Parameters
		"""
		self.screen_dim = 500
		self.screen = None

		"""
		### Network latency
		"""
		self.delay_per_level   = 1 				# 5s
		self.max_network_level = 1		# The network strenght is an integer from 0 to max_network_level
		self.delay_handler_rel_dis  = DelayHandler(self.delt, self.delay_per_level, self.max_network_level)
		self.delay_handler_rel_vel  = DelayHandler(self.delt, self.delay_per_level, self.max_network_level)
		self.delay_handler_idm_act  = DelayHandler(self.delt, self.delay_per_level, self.max_network_level)

		"""
		### Initialziation
		"""
		self.InitializeEnvironmentVariables()

	def InitializeEnvironmentVariables(self):
		self.episode_steps = 0
		self.done = False

		"""
		### Initial conditions
		"""
		### Front vehicle
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.InitializeFvVel()
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)
		self.fv_acc_list = InputAccGenerator(self.max_episode_steps, self.delt, self.fv_init_vel, 
											 self.fv_min_vel, self.fv_max_acc, self.fv_max_acc, 
											 mode = 'SplineProfile')

		### Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32) 
		self.prev_acc     = 0.0
		
		rel_pose = self.fv_state - self.ego_state
		
		### Classical control
		self.model_IDM_1 = IDM()
		self.model_IDM_2 = IDM()

		action_IDM_1 = self.model_IDM_1.action(rel_pose[0], -rel_pose[1], self.ego_state[1])
		action_IDM_1 = np.clip(action_IDM_1, -self.max_acc, self.max_acc)
		action_IDM_2 = self.model_IDM_2.action(rel_pose[0], -rel_pose[1], self.ego_state[1])
		action_IDM_2 = np.clip(action_IDM_2, -self.max_acc, self.max_acc)

		### Network latency
		# self.ns_ts = NetworkStrength(self.max_episode_steps, n_min = 0, n_max = 2, low = 0, high = self.max_network_level)
		self.ns_ts = NetworkStrength(self.max_episode_steps, n_min = 0, n_max = 2, low = 0, high = 0)
		self.delay_handler_rel_dis.reset()
		self.delay_handler_rel_vel.reset()
		self.delay_handler_idm_act.reset()

		### State value
		self.state = np.append(rel_pose,   action_IDM_1)
		self.state = np.append(self.state, rel_pose)
		self.state = np.append(self.state, action_IDM_2)
		self.state = np.append(self.state, self.prev_acc)
		self.state = np.append(self.state, self.ns_ts[0])

	def InitializeFvPos(self):
		if self.train:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		else:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m

	def InitializeFvVel(self):
		if self.train:
			return 0
		else:
			return min(self.fv_max_vel, max(self.fv_min_vel + np.random.rand()*(self.fv_max_vel - self.fv_min_vel), 
											self.fv_min_vel))	# (20 +-5)m/s or 60mph

	def InitializeEgoPos(self):
		return 0

	def InitializeEgoVel(self):
		if self.train:
			return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s
		else:
			return min(self.fv_max_vel, max(self.fv_min_vel + np.random.rand()*(self.fv_max_vel - self.fv_min_vel), 
											self.fv_min_vel))


	def step(self, action):
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1] 

		"""
		### Front vehicle state transition
		"""
		### Acceleration input to the front vehicle
		fv_acc = self.fv_acc_list[self.episode_steps]
		if self.train:
			fv_acc = 0.0
		else:
			fv_acc = fv_acc*self.fv_max_acc
		
		### Clipping acceleration to keep within velocity limits
		if fv_vel >= self.fv_max_vel:
			if fv_vel + fv_acc*self.delt >= self.fv_max_vel:
				fv_acc = 0.0
		else:
			if fv_vel + fv_acc*self.delt >= self.fv_max_vel:
				fv_acc = (self.fv_max_vel - fv_vel)/self.delt
		
		if fv_vel <= self.fv_min_vel:
			if fv_vel + fv_acc*self.delt <= self.fv_min_vel:
				fv_acc = 0.0
		else:
			if fv_vel + fv_acc*self.delt <= self.fv_min_vel:
				fv_acc = (self.fv_min_vel - fv_vel)/self.delt		

		### State update
		fv_pos = fv_pos + fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		fv_vel = fv_vel + fv_acc*self.delt
		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		
		"""
		### Ego vehicle state transition
		"""
		### Acceleration input to the ego vehicle
		action = np.clip(action, self.action_low, self.action_high)[0]
		ego_acc = action.item()*self.max_acc

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

		"""
		Network latency
		"""
		ns = self.ns_ts[self.episode_steps]

		"""
		# Reward function
		"""
		### Reward for moving forward
		reward = ego_dist_trav/self.ego_max_dist
		
		### Jerk cost function
		jerk = abs(ego_acc - self.prev_acc)
		self.prev_acc = ego_acc
		# reward -= self.jerk_cost_coef*jerk

		### Reward for being too close to the front vehicle
		rel_dis = fv_pos - ego_pos
		if rel_dis < self.safety_dist:
			reward += self.violating_safety_dist_reward

		"""
		# Noise corruption
		"""
		rel_pose = self.fv_state - self.ego_state
		rel_pose_noisy = np.array([self.depth_noise_model(rel_pose[0]), 
								   self.vel_noise_model(rel_pose[1], rel_pose[0])]).flatten()
		rel_pose_laggy = np.array([self.delay_handler_rel_dis.update(rel_pose[0], ns), 
								   self.delay_handler_rel_vel.update(rel_pose[1], ns)]).flatten()
		# print(f'ns: {ns}')
		# print(f'rel_dis: {rel_pose[0]}, rel_dis_laggy: {rel_pose_laggy[0]}')
		# print(f'rel_vel: {rel_pose[1]}, rel_vel_laggy: {rel_pose_laggy[1]}')

		"""
		### Classical control
		"""
		action_IDM_1 = self.model_IDM_1.action(rel_pose_noisy[0], -rel_pose_noisy[1], self.ego_state[1])
		action_IDM_1 = np.clip(action_IDM_1, -self.max_acc, self.max_acc)
		action_IDM_2 = self.model_IDM_2.action(rel_pose_laggy[0], -rel_pose_laggy[1], self.ego_state[1])
		action_IDM_2 = np.clip(action_IDM_2, -self.max_acc, self.max_acc)
		action_IDM_21 = self.delay_handler_idm_act.update(action_IDM_2, ns)
		# print(f'idm_act: {action_IDM_2}, idm_act_laggy: {action_IDM_21}')
		action_IDM_2 = action_IDM_21

		self.state = np.append(rel_pose_noisy, action_IDM_1)
		self.state = np.append(self.state,     rel_pose_laggy)
		self.state = np.append(self.state, 	   action_IDM_2)
		self.state = np.append(self.state, 	   self.prev_acc)
		self.state = np.append(self.state, 	   ns)
		
		"""
		### Observation
		"""
		obs = self.state.copy()

		"""
		### Environment handling 
		"""
		### Terminating the episode
		if rel_dis <= 2 or self.episode_steps >= self.max_episode_steps:
			# print("distance remaining : ", rel_dis)
			self.done = True 

		self.episode_steps += 1

		info = {
			"fv_pos"  : fv_pos,
			"fv_vel"  : fv_vel,
			"fv_acc"  : fv_acc, 
			"ego_pos" : ego_pos,
			"ego_vel" : ego_vel,
			"ego_acc" : ego_acc,
			"idm_1"	  : action_IDM_1,
			"idm_2"	  : action_IDM_2,
            "dis_rem" : self.state[0],
		}

		return obs, reward, self.done, info

	def reset(self, seed=0):
		if not self.train:
			np.random.seed(seed)
		self.InitializeEnvironmentVariables()

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