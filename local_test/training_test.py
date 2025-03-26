from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SnakeGameEnv


class LRSchedule:
    def __init__(self, total_timesteps, decay_start=0.5):
        self.total_timesteps = total_timesteps
        self.decay_start = 1 - decay_start
        self.timesteps_trained = 0

    def __call__(self, _):
        self.timesteps_trained += 1
        progress_remaining = 1 - self.timesteps_trained / self.total_timesteps

        if progress_remaining < self.decay_start:
            return 3e-4 * (progress_remaining / self.decay_start)
        return 3e-4 * progress_remaining


log_dir = "logs"

# vec_env = make_vec_env(lambda: SnakeGameEnv(num_snakes=1, num_teams=1), n_envs=2)
env = SnakeGameEnv(num_snakes=1, num_teams=1)


model = PPO('MultiInputPolicy', env, verbose=True, device='cuda', tensorboard_log=log_dir, n_steps=1024, learning_rate=LRSchedule(10000, 0.3))
# # model = PPO.load("ppo_snake", env=env, device="cuda", tensorboard_log=log_dir)


for i in range(10):
    model.learn(10000, progress_bar=True, tb_log_name="test", reset_num_timesteps=False)
    model.save('test.zip')