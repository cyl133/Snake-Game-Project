from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
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
                # Get info from the environment
                env = self.training_env.envs[0]  # Assuming single environment or all envs are the same
                
                # Calculate food eaten (tail size increase from initial)
                current_tail_size = env.env.snakes[0].tail_size if hasattr(env, 'env') and hasattr(env.env, 'snakes') else 0
                food_eaten = max(0, current_tail_size - env.initial_tail_size)
                
                # Get episode length
                episode_length = env.env.time_steps if hasattr(env, 'env') else 0
                
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
        "n_steps": 1024,
        "total_timesteps": 500000,  # 50 * 10000
        "decay_start": 0.3,
    },
    sync_tensorboard=True,  # Auto-upload tensorboard metrics
)

log_dir = "logs"

# Create environment
env = make_vec_env(SnakeGameEnv, env_kwargs={"num_snakes": 1, "num_teams": 1})

# Create model with wandb callback
model = PPO(
    'MultiInputPolicy', 
    env, 
    verbose=True, 
    device='cuda', 
    tensorboard_log=log_dir, 
    n_steps=1024
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

# Training loop
total_timesteps = 0
for i in range(50):
    model.learn(
        10000, 
        progress_bar=True, 
        tb_log_name="test", 
        reset_num_timesteps=False,
        callback=callbacks
    )
    model.save(f'models/{run.id}/checkpoint_{i}')
    total_timesteps += 10000
    
    # Log custom metrics at each checkpoint
    wandb.log({
        "checkpoint": i,
        "total_timesteps": total_timesteps,
        "avg_food_eaten_last_checkpoint": np.mean(snake_metrics_callback.food_eaten[-100:]) if snake_metrics_callback.food_eaten else 0,
        "avg_episode_length_last_checkpoint": np.mean(snake_metrics_callback.episode_lengths[-100:]) if snake_metrics_callback.episode_lengths else 0,
    })

# Close wandb run when done
wandb.finish()