from matplotlib.pyplot import axes
import numpy as np
import matplotlib.pyplot as plt

import random as pyrandom

class PiecewiseLinearProfile():
    def __init__(self, N, n_min, n_max):
        self.N     = N
        self.n_max = n_max
        self.n_min = n_min

    def generate(self):
        n = np.random.randint(self.n_min, self.n_max + 1)
        
        anchor_x = np.array(pyrandom.sample(range(self.N), n))
        anchor_x.sort()
        anchor_x = np.insert(anchor_x, 0, 0)
        anchor_x = np.append(anchor_x, self.N)
        
        anchor_y = [np.random.uniform()]
        for i in range(len(anchor_x)-1):
            s = 3*(anchor_x[i+1] - anchor_x[i])/self.N
            y_min = max(anchor_y[-1] + -1*s, -1)
            y_max = min(anchor_y[-1] +  1*s,  1)
            anchor_y.append(np.random.uniform(y_min, y_max))
        
        anchor_y = np.array(anchor_y)
        

        X = np.arange(0, self.N + 1)
        Y = np.interp(X, anchor_x, anchor_y)

        return(Y)

# profile_gen = PiecewiseLinearProfile(100, 1, 5)

# fig, axes = plt.subplots(1,1)
# axes.set_ylim((-1,1))
# plt.plot(profile_gen.generate())

# plt.show()