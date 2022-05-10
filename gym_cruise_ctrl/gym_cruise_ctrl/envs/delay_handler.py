import numpy as np
from gym_cruise_ctrl.envs.network_strength import NetworkStrength
import matplotlib.pyplot as plt

class DelayHandler():
    def __init__(self, sampling_time, delay_per_level, max_level):
        self.buffer_len = delay_per_level/sampling_time*max_level + 1
        self.max_level  = max_level
        self.buffer     = np.zeros(int(self.buffer_len))
        self.buffer_ind = self.buffer_len - 1

    def update(self, sig, ns):
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = sig
        
        self.buffer_ind = int(max(ns*(self.buffer_len - 1)/self.max_level, self.buffer_ind - 1))
        sig_m = self.buffer[self.buffer_ind]

        return sig_m

    def reset(self):
        self.buffer     = np.zeros(int(self.buffer_len))
        self.buffer_ind = self.buffer_len - 1

        
# N = 50

# ns = NetworkStrength(N, n_min = 0, n_max = 3, low = 1, high = 1)
# sig = np.arange(N+1)

# max_delay = 5
# t = 1

# delay_handler = DelayHandler(1, 5, 1)
# delay_handler.reset()

# sig_measured = []
# for i in range(N+1):
#     sig_measured.append(delay_handler.update(sig[i], ns[i]))


# fig, axes = plt.subplots(2,1)
# axes[0].set_ylim((-1,6))
# axes[0].plot(ns)

# axes[1].plot(sig, label = 'actual')
# axes[1].plot(sig_measured, label = 'measured')

# plt.savefig('img.png')
# plt.show()