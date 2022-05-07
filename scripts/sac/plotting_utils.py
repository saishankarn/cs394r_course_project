import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
import datetime as datetime

def PlotTrainResults(log_folder, title='Learning Curve'):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    print(x,y)

    # def moving_average(values, window):
    #     """
    #     Smooth values by doing a moving average
    #     :param values: (numpy array)
    #     :param window: (int)
    #     :return: (numpy array)
    #     """
    #     weights = np.repeat(1.0, window) / window
    #     return np.convolve(values, weights, 'valid')
    #y = moving_average(y, window=50)
    # Truncate x
    #x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x[:50000], y[:50000])
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)

    plot_dir = 'plots/train_plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir+'/fig_'+datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')+'.png')
    fig.tight_layout()
    plt.show()

class PlotTestResults():
    def __init__(self) -> None:        
        self.reward_to_go_list = [0]
        
        self.rel_dis_list = []
        self.rel_vel_list = []

        self.fv_pos_list  = []
        self.fv_vel_list  = []
        self.fv_acc_list  = []
        self.ego_pos_list = []
        self.ego_vel_list = []
        self.ego_acc_list = []
    
    def store(self, obs, reward, info):
        self.reward_to_go_list.append(self.reward_to_go_list[-1] + reward)
        self.rel_dis_list.append(obs[0])
        self.rel_vel_list.append(obs[1])

        self.fv_pos_list.append(info["fv_pos"])
        self.fv_vel_list.append(info["fv_vel"])
        self.fv_acc_list.append(info["fv_acc"])

        self.ego_pos_list.append(info["ego_pos"])
        self.ego_vel_list.append(info["ego_vel"])
        self.ego_acc_list.append(info["ego_acc"])

    def plot(self):
        if len(self.reward_to_go_list) > len(self.fv_pos_list):
            del self.reward_to_go_list[0]
        
        fig, axes = plt.subplots(2,3, figsize = (20,10))

        axes[0,0].plot(self.rel_dis_list)
        axes[0,1].plot(self.rel_vel_list)
        axes[0,2].plot(self.reward_to_go_list)

        axes[1,0].plot(self.fv_pos_list, color = 'b', label = 'Front vehicle')
        axes[1,0].plot(self.ego_pos_list, color = 'r',  label = 'Ego vehicle')

        axes[1,1].plot(self.fv_vel_list, color = 'b', label = 'Front vehicle')
        axes[1,1].plot(self.ego_vel_list, color = 'r',  label = 'Ego vehicle')

        axes[1,2].plot(self.fv_acc_list, color = 'b', label = 'Front vehicle')
        axes[1,2].plot(self.ego_acc_list, color = 'r',  label = 'Ego vehicle')
        
        axes[1,0].legend()
        axes[1,1].legend()
        axes[1,2].legend()
        
        axes[0,0].title.set_text('Relative distance between vehicles')
        axes[0,1].title.set_text('Relative velocity between vehicles')
        axes[0,2].title.set_text('Reward to go over time')
        axes[1,0].title.set_text('Positions of front and ego vehicles')
        axes[1,1].title.set_text('Velocities of front and ego vehicles')
        axes[1,2].title.set_text('Accelerations of front and ego vehicles')

        axes[1,0].set_xlabel('Time steps')
        axes[1,1].set_xlabel('Time steps')
        axes[1,2].set_xlabel('Time steps')

        axes[0,0].set_ylabel('Distance (m)')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,2].set_ylabel('Total reward')
        axes[1,0].set_ylabel('Position (m)')
        axes[1,1].set_ylabel('Velocity (m/s)')
        axes[1,2].set_ylabel('Acceleration (m/s)')

        axes[0,0].set_xlim([0, 1000])
        axes[0,1].set_xlim([0, 1000])
        axes[0,2].set_xlim([0, 1000])
        axes[1,0].set_xlim([0, 1000])
        axes[1,1].set_xlim([0, 1000])
        axes[1,2].set_xlim([0, 1000])

        plot_dir = 'plots/test_plots'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_dir+'/fig'+'.png')
        # plt.savefig(plot_dir+'/fig_'+datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')+'.png')
        fig.tight_layout()
        plt.show()