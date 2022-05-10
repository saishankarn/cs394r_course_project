import os 
import argparse
 
from sac import SoftActorCritic
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--log_dir", type=str, default='./tmp/sac/pure_delay_1s_2')
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--modify_alpha_after", type=int, default=100000)
    parser.add_argument("--num_test_episodes", type=int, default=1000)
    
    args = parser.parse_args()
     
    os.makedirs(args.log_dir, exist_ok=True)
    
    env_name = 'cruise-ctrl-v4'
    soft_actor_critic = SoftActorCritic(env_name)

    policy_path = os.path.join(args.log_dir, 'own_sac_best_policy800000.pt')
    
    episode_rewards, distance_remaining = soft_actor_critic.test(policy_path, args)
    mean_episode_reward = sum(episode_rewards) / args.num_test_episodes
    print(mean_episode_reward)


    reduced_distance_remaining = [i for i in distance_remaining if i > 5]
    try:
        mean_distance_remaining = sum(reduced_distance_remaining) / len(reduced_distance_remaining)
        print(mean_distance_remaining)
    except: 
        pass

    lesser_than_5 = [i for i in distance_remaining if i < 5 and i > 2]
    try:
        mean_distance_remaining = sum(lesser_than_5) / len(lesser_than_5)
        print(mean_distance_remaining)
    except:
        pass

    lesser_than_2 = [i for i in distance_remaining if i <= 2]
    try:
        mean_distance_remaining = sum(lesser_than_2) / len(lesser_than_2)
        print(mean_distance_remaining)
    except:
        pass

    print(len(reduced_distance_remaining))
    print(len(lesser_than_5))
    print(len(lesser_than_2))

    soft_actor_critic.visualize(policy_path, args)