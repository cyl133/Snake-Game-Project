from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_env import SnakeGameEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Custom CNN for Dict observation spaces with small images
class SnakeFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for Dict observation space with 'image' and 'vector' keys.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        # Extract spaces
        self.image_space = observation_space.spaces['image']
        self.vector_space = observation_space.spaces['vector']
        
        # Image input has 3 channels (RGB)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output shape
        with th.no_grad():
            # Sample image is in [H, W, C] format from gym_env
            sample = self.image_space.sample()
            # Need to convert to [1, C, H, W] for PyTorch CNN
            sample_tensor = th.as_tensor(sample).float().permute(2, 0, 1).unsqueeze(0)
            n_flatten = self.cnn(sample_tensor).shape[1]
        
        # Vector features
        vector_dim = self.vector_space.shape[0]
        
        # Linear layers for combining features
        self.linear_cnn = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
        self.linear_vector = nn.Sequential(nn.Linear(vector_dim, 16), nn.ReLU())
        
        # Final combined output
        self.linear_combined = nn.Sequential(
            nn.Linear(64 + 16, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        # Process image observations
        image_obs = th.as_tensor(observations['image']).float()
        
        # Convert from [batch, height, width, channels] to [batch, channels, height, width]
        # The image from gym_env is in [H, W, C] format, so we need correct permutation
        image_obs = image_obs.permute(0, 3, 1, 2)
        
        image_features = self.cnn(image_obs)
        image_features = self.linear_cnn(image_features)
        
        # Process vector observations
        vector_obs = th.as_tensor(observations['vector']).float()
        vector_features = self.linear_vector(vector_obs)
        
        # Combine features
        combined = th.cat([image_features, vector_features], dim=1)
        return self.linear_combined(combined)


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


def main():
    # Initialize wandb
    run = wandb.init(
        project="snake-rl",  # Project name
        config={
            "algorithm": "PPO",
            "num_snakes": 1,
            "num_teams": 1,
            "n_steps": 256,
            "total_timesteps": 500000,
            "n_envs": 8,
            "batch_size": 256,
        },
        sync_tensorboard=True,  # Auto-upload tensorboard metrics
    )

    log_dir = "logs"

    # Create environment with parallel processing
    env = make_vec_env(
        SnakeGameEnv, 
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"num_snakes": 1, "num_teams": 1}
    )

    # Configure a custom policy with our feature extractor
    policy_kwargs = {
        "features_extractor_class": SnakeFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
    }

    # Create model with appropriate parameters for parallel environments
    model = PPO(
        'MultiInputPolicy', 
        env, 
        verbose=1, 
        device='cuda', 
        tensorboard_log=log_dir, 
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs  # Use our custom features extractor
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


if __name__ == "__main__":
    main()