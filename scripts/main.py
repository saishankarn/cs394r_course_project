# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl 

from stable_baselines3 import PPO
 
env = gym.make('cruise-ctrl-v0')
s, _ = env.reset()

num_episodes = 1000 

model = PPO("MlpPolicy", env, verbose=1, gamma = 0.99)

model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(100):
    print("state ", obs)
    action, _states = model.predict(obs, deterministic=True)
    action = [1]
    obs, reward, done, info = env.step(action)
    print("action and reward ", action, reward)
    #env.render()