import gym
import datetime as dt
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from env.erp_gym import ERP

import pandas as pd
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: ERP()])

model = PPO("MlpPolicy", env, verbose=1,learning_rate=1e-4)
model.learn(total_timesteps=100000)
model.save("ppo_erp")
tmp_path = "~/erp"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
for _ in range(1000):
        obs = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print('mean_reward',mean_reward)
        env.render()
