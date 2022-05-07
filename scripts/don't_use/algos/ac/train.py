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
    polyak = 0.995
    gamma = 0.99
    buffer_size = 1e6
    total_steps = 1e6
    update_after = 1e3 
    update_every = 50
    batch_size = 256     

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
    
    #Target = deepcopy(ActorCritic)  

    """
    ### Defining the parameters for training
    """
    #critic_parameters = itertools.chain(ActorCritic.critic1.parameters(), ActorCritic.critic2.parameters())
    critic_parameters = ActorCritic.critic.parameters()
    critic_optimizer = Adam(critic_parameters, lr=learning_rate)

    policy_parameters = ActorCritic.policy.parameters()
    policy_optimizer = Adam(policy_parameters, lr=learning_rate)

    # The Target network's weights are not updated through backpropogation
    # The Target network's weights are only updated through polyak averaging
    # Hence setting the requires_grad of the target network parameters to false
    #for param in Target.parameters():
    #    param.requires_grad = False 

    """
    ### Initializing the replay buffer
    """
    replay_buffer = ReplayBuffer(state_space=env.observation_space, action_space=env.action_space, buffer_size=buffer_size)

    """
    ### Functions for policy and critic loss
    """

    # Calculating the action value loss
    def action_value_loss(data):
        state, action, reward, next_state, done = data['state'], data['action'], data['reward'], data['next_state'], data['done']
        # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        
        # Calculating the current (s,a)'s Q value to compare against a bootstrapped target
        q_value = ActorCritic.critic(state, action) # Calculating Q(s,a) from the actor critic's critic1 network
        #q_value2 = ActorCritic.critic2(state, action) # Calculating Q(s,a) from the actor critic's critic2 network
        # print(q_value1.shape, q_value2.shape)

        # using torch.no_grad() every time when we don't want to compute the gradients and just update the values
        with torch.no_grad(): 
            #print("here at tprch no grad")
            next_action, logprob_next_action = ActorCritic.policy(next_state) # finding out the next state's action (s',a')
            #print(logprob_next_action.shape, next_action.shape)
            next_q_value = ActorCritic.critic(next_state, next_action) # Calculating Q(s',a') from the target's critic1 network
            #next_q_value2 = Target.critic2(next_state, next_action) # Calculating Q(s',a') from the target's critic2 network
            #next_q_value = torch.min(next_q_value1, next_q_value2)  # Q(s',a')
            #print(next_q_value.shape, logprob_next_action.shape)

            # Q(s,a)'s bootstrapped estimate -> r + gamma * Q(s',a')
            bootstrapped_target = reward + gamma * (1 - done) * (next_q_value)# - alpha * logprob_next_action)

        # print(q_value1.shape, bootstrapped_target.shape)
        critic_loss = torch.square(q_value - bootstrapped_target).mean() #+ torch.square(q_value2 - bootstrapped_target).mean()

        return critic_loss 

    # Calculating the policy loss
    def policy_loss(data): 
        state = data['state']
        #print("here at the policy loss function")
        action, logprob_action = ActorCritic.policy(state) # using the policy network to obtain the action for the corresponding state
        # getting the Q(s,a) for the state action pair using both the critic networks
        q_value = ActorCritic.critic(state, action)
        #q_value2 = ActorCritic.critic2(state, action)        
        #q_value = torch.min(q_value1, q_value2) # Q(s,a)
        #print(q_value.requires_grad)
        # policy loss
        # intuitive understanding - 
        # for the policy loss to reduce, the q_value should increase, i.e. the policy network should output an action distribution
        # which results in the highest state action value 
        # Also, for the policy loss to reduce, the action's log probability should decrease, i.e. the policy must become more exploratory.
        # print(logprob_action.shape, q_value.shape)
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

        # Unfreeze Q-networks so you can optimize it at next step.
        for parameter in critic_parameters:
            parameter.requires_grad = True

        # Finally, update target networks by polyak averaging.
        #with torch.no_grad():
        #    for param, target_param in zip(ActorCritic.parameters(), Target.parameters()):
        #        target_param.data.mul_(polyak)
        #        target_param.data.add_((1 - polyak) * param.data) 

        return critic_loss, policy_loss_val

    def get_action(state, deterministic=False):
        # print(state.shape)
        return ActorCritic.act(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic) 

    def test_agent(test_env): 
        random_seeds = np.random.choice(10000, size=(num_test_episodes,))
        
        test_episodes_returns = []
        for test_ep_idx in range(num_test_episodes):
            state = test_env.reset(seed=random_seeds[test_ep_idx])
            done = False 
            episode_return = 0

            while not done: 
                state_tensor = torch.tensor(state).unsqueeze(0)
                state, reward, done, _ = test_env.step(get_action(state_tensor, deterministic=True).squeeze(0))
                episode_return += reward

            test_episodes_returns.append(episode_return)

        return test_episodes_returns 

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
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        state_tensor = torch.tensor(state).unsqueeze(0)
        #print(state.shape)
        action = get_action(state_tensor).squeeze(0)
        #print(action.shape, "---")

        # Step the env
        next_state, reward, done, _ = env.step(action) 
        rewards.append(reward)

        # writing the reward summary
        write_reward_summary(writer, rewards, t)

        ep_ret += reward

        # Store experience to replay buffer
        # state shape - (3,)
        # action shape - (1,)
        replay_buffer.store(state, action, reward, next_state, done)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        state = next_state

        # End of trajectory handling
        if done:
            print("episode ended, return : ", ep_ret)
            state, ep_ret = env.reset(), 0

        # Update handling 
        if t >= update_after and t % update_every == 0: 
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                critic_loss, policy_loss_val = update(data=batch)  

                # loss summary writing
                write_loss_summary(writer, critic_loss, policy_loss_val, rewards, num_updates)
                num_updates += 1
                

            # Saving the best model
            save_path = os.path.join(args.log_dir, 'own_sac_best_model.pt')


            mean_reward = np.mean(rewards[-100:])            
            if mean_reward > best_mean_reward:
                print("*****************************************************************************")
                print("saving a better model")
                print("*****************************************************************************")
                best_mean_reward = mean_reward
                torch.save(ActorCritic.state_dict(), save_path) 

            #if t > 100000:
            #    returns = test_agent(test_env) 
            #    print("---------------------------------------------------------")
            #    print("The returns : ", returns)

        """
        ### reducing the entropy coefficient over time
        """
        if t % 100000 == 0:
            print("*****************************************************************************")
            print("improving apha OR reducing entropy coefficient")
            print("*****************************************************************************")
            alpha /= 2


    """
    ### saving the final model
    """
    save_path = os.path.join(args.log_dir, 'own_sac_last_model.pt')
    torch.save(ActorCritic.state_dict(), save_path) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/tmp/sac/sac_for_basic", help="logging directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)