from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SnakeGameEnv


log_dir = "logs"

vec_env = make_vec_env(lambda: SnakeGameEnv(num_snakes=1, num_teams=1), n_envs=3)
# env = SnakeGameEnv(num_snakes=1, num_teams=1)


model = PPO('MultiInputPolicy', vec_env, verbose=True, device='cuda', tensorboard_log=log_dir, n_steps=1024, )
# # model = PPO.load("ppo_snake", env=env, device="cuda", tensorboard_log=log_dir)


for i in range(10):
    model.learn(100000, progress_bar=True, tb_log_name="PPO-3.132", reset_num_timesteps=False)
    model.save('PPO-3.132.zip')