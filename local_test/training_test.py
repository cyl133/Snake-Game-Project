from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SnakeGameEnv
import wandb
from wandb.integration.sb3 import WandbCallback


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
        "learning_rate": "3e-4 with decay",
        "n_steps": 1024,
        "total_timesteps": 500000,  # 50 * 10000
        "decay_start": 0.3,
    },
    sync_tensorboard=True,  # Auto-upload tensorboard metrics
)

log_dir = "logs"

# Create environment
env = SnakeGameEnv(num_snakes=1, num_teams=1)

# Create model with wandb callback
model = PPO(
    'MultiInputPolicy', 
    env, 
    verbose=True, 
    device='cuda', 
    tensorboard_log=log_dir, 
    n_steps=1024, 
    learning_rate=LRSchedule(10000, 0.3)
)

# Create wandb callback
wandb_callback = WandbCallback(
    gradient_save_freq=1024,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

# Training loop
total_timesteps = 0
for i in range(50):
    model.learn(
        10000, 
        progress_bar=True, 
        tb_log_name="test", 
        reset_num_timesteps=False,
        callback=wandb_callback
    )
    model.save(f'models/{run.id}/checkpoint_{i}')
    total_timesteps += 10000
    
    # Log custom metrics at each checkpoint
    wandb.log({
        "checkpoint": i,
        "total_timesteps": total_timesteps,
    })

# Close wandb run when done
wandb.finish()