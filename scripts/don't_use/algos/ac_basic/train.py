# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl
import numpy as np
import os 
import sys
import argparse
import itertools
from copy import deepcopy

import torch
from torch.optim import Adam
 
from models import ActorCriticNetwork 
from buffer import ReplayBuffer 

from torch.utils.tensorboard import SummaryWriter

def train(args):

    """
    ### listing all the training hyperparameters here 
    """
    alpha = 0.02 # entropy coefficient
    learning_rate = 3e-4 # learning rate
    buffer_size = 1e6 
    total_steps = 1e6
    gamma = 0.99

    num_test_episodes = 10

    device = torch.device("cuda")
    """
    ### Initialize the environment
    """
    env = gym.make('cruise-ctrl-v0', train=True, noise_required=False) 
    state_space = env.observation_space
    action_space = env.action_space 
    
    test_env = gym.make('cruise-ctrl-v0', train=False, noise_required=False)

    """
    ### Instantiating the neural network policies and value function estimators 
    ### We have an actor critic and a target actor critic which helps in reducing the maximization bias problem
    """
    ActorCritic = ActorCriticNetwork(state_space, action_space, device)
    #ActorCritic.load_state_dict(torch.load(os.path.join(args.log_dir, 'own_sac_best_model.pt')))
    
    """
    ### Defining the parameters for training
    """
    critic_parameters = ActorCritic.critic.parameters()
    critic_optimizer = Adam(critic_parameters, lr=learning_rate)

    policy_parameters = ActorCritic.policy.parameters()
    policy_optimizer = Adam(policy_parameters, lr=learning_rate)

    """
    ### Functions for policy and critic loss
    """

    # Calculating the action value loss
    def action_value_and_policy_loss(state, action, next_state, reward, done):
        # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Calculating the current (s,a)'s Q value to compare against a bootstrapped target
        q_value = ActorCritic.critic(state_tensor, action_tensor) # Calculating Q(s,a) from the actor critic's network

        # using torch.no_grad() every time when we don't want to compute the gradients and just update the values
        # and we don't want the bootstrapped_target to have any gradients (semi-gradient descent)
        with torch.no_grad(): 
            next_action_tensor, logprob_next_action_tensor = ActorCritic.policy(next_state_tensor) # finding out the next state's action (s',a')
            #print(logprob_next_action.shape, next_action.shape)
            next_q_value = ActorCritic.critic(next_state_tensor, next_action_tensor) # Calculating Q(s',a') from the critic network

            # Q(s,a)'s bootstrapped estimate -> r + gamma * Q(s',a')
            bootstrapped_target = reward + gamma * (1 - done) * (next_q_value)

        # print(q_value.requires_grad, bootstrapped_target.requires_grad)
        # print(q_value1.shape, bootstrapped_target.shape)
        td_error = q_value - bootstrapped_target
        critic_loss = torch.square(td_error).mean() 

        action, logprob_action = ActorCritic.policy(state_tensor)
        #print(logprob_action.shape, td_error.shape, logprob_action, td_error)
        policy_loss = -1 * td_error.detach() * logprob_action 
        policy_loss = policy_loss.mean()
        # print(policy_loss, policy_loss.shape, critic_loss, critic_loss.shape) 

        return critic_loss, policy_loss 
 

    # updating actor-critic parameters
    def update(state, action, next_state, reward, done):
        
        critic_loss, policy_loss_val = action_value_and_policy_loss(state, action, next_state, reward, done)
        
        # First run one gradient descent step for critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for parameter in critic_parameters:
            parameter.requires_grad = False

        # Next run one gradient descent step for pi.
        policy_optimizer.zero_grad()
        policy_loss_val.backward()
        policy_optimizer.step()

        #print("critic loss : ", critic_loss)
        #print("policy loss : ", policy_loss_val)

        # Unfreeze Q-networks so you can optimize it at next step.
        for parameter in critic_parameters:
            parameter.requires_grad = True

        return critic_loss, policy_loss_val

    def get_action(state, deterministic=False):
        # print(state.shape)
        action, logprob_action = ActorCritic.act(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic) 
        return action.squeeze(0), logprob_action

    def write_loss_summary(writer, critic_loss, policy_loss_val, rewards, num_updates):
        # writing the critic loss
        writer.add_scalar('Loss/critic', critic_loss.item(), num_updates)
        # writing the policy loss
        writer.add_scalar('Loss/policy', policy_loss_val.item(), num_updates)

    def write_reward_summary(writer, rewards, num_updates):
        # writing the mean reward for the last 100 time steps 
        mean_r = np.mean(rewards[-100:]) 
        writer.add_scalar('Reward/meanReward', mean_r, num_updates)

    
    # Main loop: collect experience in env and update/log each epoch
    state, ep_ret = env.reset(), 0
    
    # to store the rewards per each time step
    # to update the best mean reward, which can be used to evaluate whether a trained model is better / worse than the existing saved model
    rewards = []
    best_mean_reward = -100  

    # to visualize the training curve on tensorboard 
    writer = SummaryWriter(args.log_dir) 
    num_updates = 0

    for t in range(int(total_steps)):
        state_tensor = torch.tensor(state).unsqueeze(0)
        action, logprob_action = get_action(state_tensor)

        # Step the env
        next_state, reward, done, _ = env.step(action) 
        rewards.append(reward)

        # writing the reward summary
        write_reward_summary(writer, rewards, t)

        ep_ret += reward

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        state = next_state

        # End of trajectory handling
        if done:
            print("episode ended, return : ", ep_ret)
            state, ep_ret = env.reset(), 0

        # updating the critic and policy networks' weights for the current time step
        critic_loss, policy_loss_val = update(state, action, next_state, reward, done)  

        # loss summary writing
        write_loss_summary(writer, critic_loss, policy_loss_val, rewards, t)

        # Saving the best model
        save_path = os.path.join(args.log_dir, 'own_sac_best_model.pt')


        mean_reward = np.mean(rewards[-100:])            
        if mean_reward > best_mean_reward:
            print("*****************************************************************************")
            print("saving a better model")
            print("*****************************************************************************")
            best_mean_reward = mean_reward
            torch.save(ActorCritic.state_dict(), save_path) 

    """
    ### saving the final model
    """
    save_path = os.path.join(args.log_dir, 'own_sac_last_model.pt')
    torch.save(ActorCritic.state_dict(), save_path) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/tmp/ac/ac_for_basic", help="logging directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)