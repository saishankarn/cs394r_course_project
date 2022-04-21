# the code tries to implement an RL agent to the cruise-ctrl-v0 env 
import gym 
import gym_cruise_ctrl 

from stable_baselines3 import PPO

env = gym.make('cruise-ctrl-v0')
s, _ = env.reset()

num_episodes = 1000 

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10)

obs = env.reset()
for i in range(100):
    print(obs)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action, reward)
    #env.render()