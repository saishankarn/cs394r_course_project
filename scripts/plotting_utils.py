import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

import sys 

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    print(x,y)
    print(len(x))
    #y = moving_average(y, window=50)
    # Truncate x
    #x = x[len(x) - len(y):]
    #print(len(x))
    #print(len(y))

    fig = plt.figure(title)
    plt.plot(x[:500], y[:500])
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.savefig('ploting_results.png')

plot_results(sys.argv[1])