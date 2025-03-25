import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SnakeGameEnv
import time


log_dir = "logs"


env = SnakeGameEnv(num_snakes=1, num_teams=1)

model = PPO('MultiInputPolicy', env, verbose=True, device='cuda', tensorboard_log=log_dir, n_steps=64)
# model = PPO.load("ppo_snake", env=env, device="cuda", tensorboard_log=log_dir)


for i in range(1):
    model.learn(64, progress_bar=True, tb_log_name="test", reset_num_timesteps=False)
    model.save('test.zip')