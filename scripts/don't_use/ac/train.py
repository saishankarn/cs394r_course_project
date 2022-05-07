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

from models import CriticNetwork, ValueNetwork, ActorCriticNetwork 
from buffer import ReplayBuffer

def train(args):

    """
    ### Initializing the environment
    """
    env = gym.make('cruise-ctrl-v0', train=True, noise_required=False) 

    """
    ### listing all the training hyperparameters here 
    """
    alpha = 0.02 # entropy coefficient
    learning_rate = 3e-4 # learning rate
    polyak = 0.995
    gamma = 0.99
    buffer_size = 1e6 
    total_steps = 1e6
    start_steps = 1e4 
    update_after = 1e3 
    update_every = 50
    batch_size = 256    

    device = torch.device("cuda")
    """
    ### Initialize the environment
    """
    env = gym.make('cruise-ctrl-v0', train=True, noise_required=False) 
    state_space = env.observation_space
    action_space = env.action_space 

    """
    ### Instantiating the neural network policies and value function estimators 
    ### We have an actor critic and a target actor critic which helps in reducing the maximization bias problem
    """
    ActorCritic = ActorCriticNetwork(state_space, action_space, device)
    Target = deepcopy(ActorCritic)

    critic_parameters = itertools.chain(ActorCritic.critic1.parameters(), ActorCritic.critic2.parameters())
    critic_optimizer = Adam(critic_parameters, lr=learning_rate)

    policy_parameters = ActorCritic.policy.parameters()
    policy_optimizer = Adam(policy_parameters, lr=learning_rate)

    # The Target network's weights are not updated through backpropogation
    # The Target network's weights are only updated through polyak averaging
    for param in Target.parameters():
        param.requires_grad = False 

    # Replay buffer
    replay_buffer = ReplayBuffer(state_space=env.observation_space, action_space=env.action_space, buffer_size=buffer_size)


    # Calculating the action value loss
    def action_value_loss(data):
        state, action, reward, next_state, done = data['state'], data['action'], data['reward'], data['next_state'], data['done']
        # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        
        # Calculating the current (s,a)'s Q value to compare against a bootstrapped target
        q_value1 = ActorCritic.critic1(state, action) # Calculating Q(s,a) from the actor critic's critic1 network
        q_value2 = ActorCritic.critic2(state, action) # Calculating Q(s,a) from the actor critic's critic2 network
        # print(q_value1.shape, q_value2.shape)

        # using torch.no_grad() every time when we don't want to compute the gradients and just update the values
        with torch.no_grad(): 
            next_action, logprob_next_action = ActorCritic.policy(next_state) # finding out the next state's action (s',a')
            #print(logprob_next_action.shape, next_action.shape)
            next_q_value1 = Target.critic1(next_state, next_action) # Calculating Q(s',a') from the target's critic1 network
            next_q_value2 = Target.critic2(next_state, next_action) # Calculating Q(s',a') from the target's critic2 network
            next_q_value = torch.min(next_q_value1, next_q_value2)  # Q(s',a')
            #print(next_q_value.shape, logprob_next_action.shape)

            # Q(s,a)'s bootstrapped estimate -> r + gamma * Q(s',a')
            bootstrapped_target = reward + gamma * (1 - done) * (next_q_value - alpha * logprob_next_action)

        critic_loss = torch.square(q_value1 - bootstrapped_target).mean() + torch.square(q_value2 - bootstrapped_target).mean()

        return critic_loss 

    # Calculating the policy loss
    def policy_loss(data):
        state = data['state']
        action, logprob_action = ActorCritic.policy(state) # using the policy network to obtain the action for the corresponding state
        
        # getting the Q(s,a) for the state action pair using both the critic networks
        q_value1 = ActorCritic.critic1(state, action)
        q_value2 = ActorCritic.critic2(state, action)        
        q_value = torch.min(q_value1, q_value2) # Q(s,a)
        
        # policy loss
        # intuitive understanding - 
        # for the policy loss to reduce, the q_value should increase, i.e. the policy network should output an action distribution
        # which results in the highest state action value 
        # Also, for the policy loss to reduce, the action's log probability should decrease, i.e. the policy must become more exploratory.
        policy_loss_val = (alpha * logprob_action - q_value).mean()

        return policy_loss_val 

    # updating actor-critic parameters
    def update(data):
        # First run one gradient descent step for critic1 and critic2
        critic_optimizer.zero_grad()
        critic_loss = action_value_loss(data)
        critic_loss.backward()
        critic_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for parameter in critic_parameters:
            parameter.requires_grad = False

        # Next run one gradient descent step for pi.
        policy_optimizer.zero_grad()
        policy_loss_val = policy_loss(data)
        policy_loss_val.backward()
        policy_optimizer.step()

        #print("critic loss : ", critic_loss)
        #print("policy loss : ", policy_loss_val)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for parameter in critic_parameters:
            parameter.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for param, target_param in zip(ActorCritic.parameters(), Target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)


    def get_action(state):
        return ActorCritic.act(torch.as_tensor(state, dtype=torch.float32))


    state, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(int(total_steps)):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            state_tensor = torch.tensor(state).unsqueeze(0)
            #print(state.shape)
            action = get_action(state_tensor)[0]
            #print(action.shape, "---")
        else:
            action = env.action_space.sample()

        # Step the env
        next_state, reward, done, _ = env.step(action)

        ep_ret += reward

        # Store experience to replay buffer
        replay_buffer.store(state, action, reward, next_state, done)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        state = next_state

        # End of trajectory handling
        if done:
            print("episode ended, return : ", ep_ret)
            state, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/tmp/sac/sac_for_basic", help="logging directory")
    
    args = parser.parse_args()
    train(args)