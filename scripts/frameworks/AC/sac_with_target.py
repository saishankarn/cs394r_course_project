import gym 
import gym_cruise_ctrl
import numpy as np
import os 
import sys
import argparse
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt

import torch

from models import ActorNetwork, CriticNetwork  
from buffer import ReplayBuffer 

from torch.utils.tensorboard import SummaryWriter

class SoftActorCriticAvecTarget():

    def __init__(self, args):
    
        """
        ### For reproducibility
        """
        torch.manual_seed(args.random_seed)

        """
        ### Initialize the environment
        """         
        self.env = gym.make(args.env_name, train=True, noise_required=args.noise_required) 
        self.test_env = gym.make(args.env_name, train=False, noise_required=args.noise_required) 
        self.state_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_action = self.action_space.high[0]

        """
        ### Initialize policy and value functions for the Soft Actor Critic
        """
        device = torch.device("cuda")
        self.policy = ActorNetwork(self.state_space, self.action_space, self.max_action, device)
        self.critic = CriticNetwork(self.state_space, self.action_space)

        self.target_critic = deepcopy(self.critic)


        # The Target network's weights are not updated through backpropogation
        # The Target network's weights are only updated through polyak averaging
        # Hence setting the requires_grad of the target network parameters to false
        
        for p in self.target_critic.parameters():
            p.requires_grad = False


    def load_weights(self, critic_path, policy_path):
        self.critic.load_state_dict(torch.load(critic1_path))
        self.target_critic.load_state_dict(torch.load(critic1_path))
        self.policy.load_state_dict(torch.load(policy_path))

    """
    ### the learning function
    """
    def learn(self, args):

        """
        ### Initializing the optimizers for policy and critic networks
        """
        
        critic_parameters = self.critic.parameters()
        critic_optimizer = torch.optim.Adam(critic_parameters, lr=args.lr)

        policy_parameters = self.policy.parameters()
        policy_optimizer = torch.optim.Adam(policy_parameters, lr=args.lr)

        """
        ### Initializing the replay buffer
        """
        replay_buffer = ReplayBuffer(state_space=self.state_space, \
                                    action_space=self.action_space, \
                                    buffer_size=args.buffer_size)

        """
        ### Initializing the iterables to be used and updated during the training 
        """

        state = self.env.reset()
        episode_return = 0 
        episode_num = 0

        # to store the rewards per each time step
        rewards = []
        best_mean_reward = -100  

        # to visualize the training curve on tensorboard 
        writer = SummaryWriter(args.log_dir) 
        num_updates = 0

        # the entropy coefficient has to be updated 
        alpha = args.alpha 

        """
        ### Training loop starts here
        """ 

        for time_step in range(args.total_steps):
         
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _ = self.policy(state_tensor, deterministic=False)
            action = action.detach().numpy().squeeze(0)

            # Step the env
            next_state, reward, done, _ = self.env.step(action) 
            rewards.append(reward)

            # logging the reward summary
            mean_r = np.mean(rewards[-100:]) 
            writer.add_scalar('Reward/meanReward', mean_r, num_updates)

            # modify the episode's return
            episode_return += reward

            # Store experience to replay buffer
            replay_buffer.store(state, action, reward, next_state, done)

            # update the current state
            state = next_state

            # What happens if the episode ends
            if done:
                print("episode ended, return : ", episode_return)
                print("distance remaining : ", state[0])

                # logging the episode return 
                writer.add_scalar('Returns/episodeReturns', episode_return, episode_num)
                writer.add_scalar('Returns/distanceRemaining', state[0], episode_num)

                state = self.env.reset()
                episode_return = 0
                episode_num += 1
    
            ###############################################################################################
            ########### Till here, we just generate rollouts and save them in the replay buffer ###########
            ###############################################################################################

            """
            ### Updating the policy's and critic's weights
            """

            # taken from spinning up RL....
            # tried Actor Critic by updating every single time step, but doesn't work 
            # so, updating only after update_every number of steps 
            # to compensate for less number of updates, we update update_every number of times 

            if time_step % args.update_every == 0:
                
                for update_idx in range(args.update_every):

                    # sampling a batch from the replay buffer
                    batch = replay_buffer.sample_batch(args.batch_size)

                    """
                    ### calculating the policy and critic loss for the sampled batch
                    ### Loss functions inspired from the spinning up RL implementation
                    """ 

                    # critic loss calculation

                    st = batch['state'] 
                    act = batch['action'] 
                    rew = batch['reward']
                    next_st = batch['next_state']
                    d = batch['done']

                    q_value = self.critic(st, act) 
         
        
                    with torch.no_grad(): 

                        next_act, logprob_next_act = self.policy(next_st)
                        next_q_value = self.target_critic(next_st, next_act) 

                    bootstrapped_target = rew + args.gamma * (1 - d) * next_q_value - alpha * logprob_next_act

                    critic_loss = torch.square(q_value - bootstrapped_target).mean()

                    # logging the critic loss
                    writer.add_scalar('Loss/critic', critic_loss.item(), num_updates)

                    # updating the critic parameters

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step() 
                
                    for p in critic_parameters:
                        p.requires_grad = False

                    # policy loss calculation

                    act, logprob_act = self.policy(st)
                    
                    q_value = self.critic(st, act)
                    
                    policy_loss = (alpha * logprob_act - q_value).mean()
                    policy_loss = policy_loss.mean()
                    
                    # logging the policy loss
                    writer.add_scalar('Loss/policy', policy_loss.item(), num_updates) 

                    # updating the policy parameters
                    
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    for p in critic_parameters:
                        p.requires_grad = True 
 
                    # update target networks by polyak averaging.
                    with torch.no_grad():
                        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                            target_param.data.mul_(args.polyak)
                            target_param.data.add_((1 - args.polyak) * param.data)

                    # incrementing the num_updates variable
                    num_updates += 1 

                # Saving the best model
                save_path_critic = os.path.join(args.log_dir, 'own_sac_best_critic.pt')
                save_path_policy = os.path.join(args.log_dir, 'own_sac_best_policy.pt')

                mean_reward = np.mean(rewards[-100:])            
                if mean_reward > best_mean_reward:
                    print("*****************************************************************************")
                    print("saving a better model")
                    print("*****************************************************************************")
                    best_mean_reward = mean_reward
                    torch.save(self.critic.state_dict(), save_path_critic)
                    torch.save(self.policy.state_dict(), save_path_policy)

                # Saving the model after every 100,000 time steps 
                save_path_critic = os.path.join(args.log_dir, 'own_sac_best_critic' + str(time_step) + '.pt')                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                save_path_policy = os.path.join(args.log_dir, 'own_sac_best_policy' + str(time_step) + '.pt')

                if time_step % 100000 == 0:
                    print("*****************************************************************************")
                    print("saving the model after ", time_step, " time steps")
                    print("*****************************************************************************")
                    best_mean_reward = mean_reward
                    torch.save(self.critic.state_dict(), save_path_critic)
                    torch.save(self.policy.state_dict(), save_path_policy)


        """
        ### saving the final model
        """
        save_path_critic = os.path.join(args.log_dir, 'own_sac_last_critic.pt')
        save_path_policy = os.path.join(args.log_dir, 'own_sac_last_policy.pt')
        torch.save(self.critic.state_dict(), save_path_critic)
        torch.save(self.policy.state_dict(), save_path_policy)

    """
    ### Testing function, to test for a fixed number of episodes
    """

    def test(self, policy_path, args):

        # loading the policy network
        self.policy.load_state_dict(torch.load(os.path.join(policy_path))) 
        
        # creating random seeds for num_test_episodes epsiodes
        random_seeds = np.random.choice(10000, size=(args.num_test_episodes,))
        
        test_episodes_returns = []

        for test_ep_idx in range(args.num_test_episodes):
            
            # resetting the environment
            state = self.test_env.reset(seed=random_seeds[test_ep_idx])
        
            done = False 
            episode_return = 0

            while not done: 

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, _ = self.policy(state_tensor, deterministic=True)
                action = action.detach().numpy().squeeze(0)

                state, reward, done, _ = self.test_env.step(action)
                episode_return += reward

            test_episodes_returns.append(episode_return)

        return test_episodes_returns 

    """
    ### Visualization function, generates the plots of position, velocity, and acceleration 
    """

    def visualize(self, policy_path, args):

        # loading the policy network
        self.policy.load_state_dict(torch.load(os.path.join(policy_path))) 
        
        # creating random seeds for num_test_episodes epsiodes
        random_seed = np.random.choice(10000)

        total_reward_list = [0]
        rel_dist_list = []
        fv_pos_list = []
        fv_vel_list = []
        fv_acc_list = []
        ego_pos_list = []
        ego_vel_list = []
        ego_acc_list = []
            
        # resetting the environment
        state = self.test_env.reset(seed=random_seed)
        
        done = False 
        episode_return = 0

        while not done: 

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _ = self.policy(state_tensor, deterministic=True)
            action = action.detach().numpy().squeeze(0)

            state, reward, done, info = self.test_env.step(action)
            episode_return += reward

            # Gather results for plotting
            total_reward_list.append(total_reward_list[-1] + reward)
            rel_dist_list.append(state[0])

            fv_pos_list.append(info["fv_pos"])
            fv_vel_list.append(info["fv_vel"])
            fv_acc_list.append(info["fv_acc"])

            ego_pos_list.append(info["ego_pos"])
            ego_vel_list.append(info["ego_vel"])
            ego_acc_list.append(info["ego_acc"]) 

        """
        ### Generate Plots
        """

        fig, axes = plt.subplots(2,3, figsize=(15,7))
        plt.rcParams.update({'font.size': 10})

        axes[0, 0].plot(total_reward_list)
        axes[0, 1].plot(rel_dist_list)
        axes[1, 0].plot(fv_pos_list, color = 'b', label = 'Front vehicle')
        axes[1, 0].plot(ego_pos_list, color = 'r',  label = 'Ego vehicle')
        axes[1, 1].plot(fv_vel_list, color = 'b', label = 'Front vehicle')
        axes[1, 1].plot(ego_vel_list, color = 'r',  label = 'Ego vehicle')
        axes[1, 2].plot(fv_acc_list, color = 'b', label = 'Front vehicle')
        axes[1, 2].plot(ego_acc_list, color = 'r',  label = 'Ego vehicle')

        axes[0, 0].title.set_text('Total reward accumulated over time')
        axes[0, 1].title.set_text('Distance between vehicles over time')
        axes[1, 0].title.set_text('Position of front and ego vehicles')
        axes[1, 1].title.set_text('Velocity of front and ego vehicles')
        axes[1, 2].title.set_text('Acceleration of front and ego vehicles')

        axes[1, 0].set_xlabel('Time steps')
        axes[1, 1].set_xlabel('Time steps')
        axes[1, 2].set_xlabel('Time steps')

        axes[0, 0].set_ylabel('Total reward')
        axes[0, 1].set_ylabel('Dist (m)')
        axes[1, 0].set_ylabel('Pos (m)')
        axes[1, 1].set_ylabel('Vel (m/s)')
        axes[1, 2].set_ylabel('Acc')

        axes[1, 0].legend()
        axes[1, 1].legend()
        axes[1, 2].legend()

        fig.tight_layout()
        plt.savefig('img.png')
        plt.show()
