import numpy as np

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