import os 
import argparse
 
from ac import ActorCritic
from sac_without_target import SoftActorCriticSansTarget
from sac_with_target import SoftActorCriticAvecTarget
from sac_with_two_critics import SoftActorCriticAvec2Critics
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--modify_alpha_after", type=int, default=100000)
    parser.add_argument("--num_test_episodes", type=int, default=1000)
    
    parser.add_argument('--env_id', type=int, default=0, help='0 for 2D, 1 for 3D and 2 for 8D state space')
    parser.add_argument("--log_dir", type=str, default='/tmp/sac/basic_sac')
    parser.add_argument("--pretrained_log_dir", type=str, default=None)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true') 
    parser.add_argument('--viz', default=False, action='store_true')
    parser.add_argument('--noise_required', default=False, action='store_true')
    parser.add_argument("--random_seed", type=int, default=0)


    
    args = parser.parse_args()

    # experimenting with only environment version v0
    args.env_name = 'cruise-ctrl-v0'
    os.makedirs(args.log_dir, exist_ok=True)

    # creating an instance for soft actor critic, the env is created along with the instance
    print(args)
    
    actor_critic = ActorCritic(args)
    soft_actor_critic_sans_target = SoftActorCriticSansTarget(args)
    soft_actor_critic_avec_target = SoftActorCriticAvecTarget(args)
    soft_actor_critic_avec_2_critics = SoftActorCriticAvec2Critics(args)
    
    basic_log_dir = args.log_dir
    
    # training the actor critic
    args.log_dir = os.path.join(basic_log_dir, 'actor_critic')  
    actor_critic.learn(args) 

    # training the soft actor critic without target
    args.log_dir = os.path.join(basic_log_dir, 'soft_actor_critic_sans_target')
    soft_actor_critic_sans_target.learn(args)

    # training the soft actor critic with target
    args.log_dir = os.path.join(basic_log_dir, 'soft_actor_critic_avec_target')
    soft_actor_critic_avec_target.learn(args)

    # training the soft actor critic with 2 critics
    args.log_dir = os.path.join(basic_log_dir, 'soft_actor_critic_avec_2_critics')
    soft_actor_critic_avec_2_critics.learn(args)