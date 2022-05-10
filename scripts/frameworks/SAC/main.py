import os 
import argparse
 
from sac import SoftActorCritic
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--modify_alpha_after", type=int, default=100000)
    parser.add_argument("--num_test_episodes", type=int, default=100)
    
    parser.add_argument('--env_id', type=int, default=0, help='0 for 2D, 1 for 3D and 2 for 8D state space')
    parser.add_argument("--log_dir", type=str, default='/tmp/sac/basic_sac')
    parser.add_argument("--pretrained_log_dir", type=str, default=None)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true') 
    parser.add_argument('--viz', default=False, action='store_true')
    parser.add_argument('--noise_required', default=False, action='store_true')
    parser.add_argument("--random_seed", type=int, default=0)


    
    args = parser.parse_args()

    # getting the right environment id

    if args.env_id == 0:
        args.env_name = 'cruise-ctrl-v0'
    elif args.env_id == 1:
        args.env_name = 'cruise-ctrl-v1'
    elif args.env_id == 2:
        args.env_name = 'cruise-ctrl-v2'    
     
    os.makedirs(args.log_dir, exist_ok=True)

    # creating an instance for soft actor critic, the env is created along with the instance
    print(args)
    
    soft_actor_critic = SoftActorCritic(args)
    
    # To load pretrained weights

    # you can load any weights (weights stored after 100,000 time steps) from this log directory by changing the name
    critic1_path = None if args.pretrained_log_dir == None else \
                    os.path.join(args.pretrained_log_dir, 'own_sac_best_critic1.pt')
    critic2_path = None if args.pretrained_log_dir == None else \
                    os.path.join(args.pretrained_log_dir, 'own_sac_best_critic2.pt')
    policy_path = None if args.pretrained_log_dir == None else \
                    os.path.join(args.pretrained_log_dir, 'own_sac_best_policy.pt')

    # loading the pretrained weights

    soft_actor_critic.load_weights(critic1_path, critic2_path, policy_path)

    # training the soft actor critic 

    if args.train:
        soft_actor_critic.learn(args)

    if args.test:
        policy_path = os.path.join(args.log_dir, 'own_sac_best_policy.pt')
        soft_actor_critic.test(policy_path, args)

    if args.viz:
        policy_path = os.path.join(args.log_dir, 'own_sac_best_policy.pt')
        soft_actor_critic.visualize(policy_path, args)