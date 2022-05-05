from matplotlib.pyplot import axes
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

import random as pyrandom

from torch import dsmm

def PiecewiseLinearProfile(N, n_min, n_max):
    n = np.random.randint(n_min, n_max + 1)
    
    #anchor_x = np.array(pyrandom.sample(range(self.N), n))
    anchor_x = np.random.choice(N, n, replace=False)
    anchor_x.sort()
    anchor_x = np.insert(anchor_x, 0, 0)
    anchor_x = np.append(anchor_x, N)

    u_min = -1
    u_max = 1
    
    anchor_y = [u_min + (u_max - u_min)*np.random.uniform()]
    for i in range(len(anchor_x)-1):
        s = (anchor_x[i+1] - anchor_x[i])/N
        y_min = max(anchor_y[-1] + u_min*s, u_min)
        y_max = min(anchor_y[-1] + u_max*s, u_max)
        anchor_y.append(np.random.uniform(y_min, y_max))
    
    anchor_y = np.array(anchor_y)
    
    X = np.arange(0, N + 1)
    Y = np.interp(X, anchor_x, anchor_y)

    return(Y)

def SplineProfile(init, N, n_min, n_max):
    k = 3
    n_min = min(k+1, n_min)
    n_max = min(k+1, n_max)
    n = np.random.randint(n_min, n_max + 1)
        
    anchor_x = np.random.choice(N, n, replace=False)
    anchor_x.sort()
    anchor_x = np.insert(anchor_x, 0, 0)
    anchor_x = np.append(anchor_x, N)
    
    anchor_y = [init]
    for i in range(len(anchor_x)-1):
        s = (anchor_x[i+1] - anchor_x[i])/N
        y_min = max(anchor_y[-1] + -1*s, 0)
        y_max = min(anchor_y[-1] +  1*s, 1)
        anchor_y.append(np.random.uniform(y_min, y_max))
    anchor_y = np.array(anchor_y)
    
    x = np.arange(0, N + 1)
    spl = UnivariateSpline(anchor_x, anchor_y, k=k)
    spld = spl.derivative()
    yd = spld(x)

    return yd

def InputAccGenerator(N, Ts, init_vel, min_vel, max_vel, max_acc, mode = 'SplineProfile'):
    if mode == 'SplineProfile':
        n_min = 4
        n_max = 8
        init = (init_vel - min_vel)/(max_vel - min_vel)
        fv_acc_list = SplineProfile(init, N, n_min, n_max)
        fv_acc_list = fv_acc_list*(max_vel - min_vel)/Ts

    elif mode == 'PiecewiseLinearProfile':
        n_min = 1
        n_max = 5
        fv_acc_list = PiecewiseLinearProfile(N, n_min, n_max)*max_acc

    else:
        raise ValueError('InputAccGenerator Mode incorrect')
    
    return fv_acc_list


# yd = InputAccGenerator(1000, 0.1, 20, 10, 30, 1, mode = 'SplineProfile')

# fig, axes = plt.subplots(1,1)
# axes.set_ylim((-1,1))
# plt.plot(yd)
# plt.savefig('img.png')
# plt.show()


        