from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_env import SnakeGameEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np


class SnakeMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = []
        self.food_eaten = []
        
    def _on_step(self):
        # Check for episode terminations
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                # Get info from the environment through infos
                infos = self.locals.get('infos', [])
                if len(infos) > idx:
                    info = infos[idx]
                    food_eaten = info.get('food_eaten', 0)
                    episode_length = info.get('episode_length', 0)
                    
                    # Log to wandb
                    wandb.log({
                        "food_eaten_per_episode": food_eaten,
                        "episode_length": episode_length
                    })
                    
                    self.episode_lengths.append(episode_length)
                    self.food_eaten.append(food_eaten)
                
        return True


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
        return 3e-4


# Initialize wandb
run = wandb.init(
    project="snake-rl",  # Project name
    config={
        "algorithm": "PPO",
        "num_snakes": 1,
        "num_teams": 1,
        "n_steps": 256,          # Reduced from 1024 since we'll collect from multiple envs
        "total_timesteps": 500000,
        "n_envs": 8,             # Number of parallel environments
        "batch_size": 256,       # Increased for better GPU utilization
    },
    sync_tensorboard=True,  # Auto-upload tensorboard metrics
)


log_dir = "logs"

# Create environment with parallel processing
env = make_vec_env(
    SnakeGameEnv, 
    n_envs=8,               # Run 8 environments in parallel
    vec_env_cls=SubprocVecEnv,  # Use subprocess vectorization for true parallelism
    env_kwargs={"num_snakes": 1, "num_teams": 1}
)

# Create model with appropriate parameters for parallel environments
model = PPO(
    'MultiInputPolicy', 
    env, 
    verbose=1, 
    device='cuda', 
    tensorboard_log=log_dir, 
    n_steps=256,           # Collect fewer steps per environment since we have multiple
    batch_size=256,        # Larger batches for better GPU utilization
    n_epochs=10,           # More optimization epochs per update
    learning_rate=3e-4,    # Standard learning rate for PPO
    clip_range=0.2,        # Standard clip range for PPO
    ent_coef=0.01          # Slightly higher entropy coefficient for exploration
)

# Create callbacks
snake_metrics_callback = SnakeMetricsCallback()
wandb_callback = WandbCallback(
    gradient_save_freq=1024,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

# Combine callbacks
callbacks = [wandb_callback, snake_metrics_callback]

# Training loop with checkpoints
total_timesteps = 0
checkpoint_frequency = 100000
for i in range(5):  # 5 training phases of 100k steps each
    model.learn(
        checkpoint_frequency, 
        progress_bar=True, 
        tb_log_name="parallel_training", 
        reset_num_timesteps=False,
        callback=callbacks
    )
    model.save(f'models/{run.id}/checkpoint_{i}')
    total_timesteps += checkpoint_frequency

# Close wandb run when done
wandb.finish()