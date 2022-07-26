import gym
import datetime as dt

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from env.erp_gym import ERP

import pandas as pd
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: ERP()])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    obs = env.reset()
    for i in range(200):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done == True:
            obs = env.reset()
            break
    
