import os 
import argparse
 
from ac import ActorCritic
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--log_dir", type=str, default='/tmp/ac/basic_sac')
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--modify_alpha_after", type=int, default=100000)
    parser.add_argument("--num_test_episodes", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=0)
    
    args = parser.parse_args()
     
    os.makedirs(args.log_dir, exist_ok=True)
    
    env_name = 'cruise-ctrl-v5'
    actor_critic = ActorCritic(env_name, args.random_seed)

    actor_critic.learn(args)

