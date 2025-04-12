import json
import os
# ssh e0694448@xlogin.comp.nus.edu 
import torch as th
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback

# Import our custom components
from feature_extractor import CustomCNN
from gym_env import SnakeGameEnv
from llm_reward_shaper import metrics_collector

# --- Configuration ---
config_dir = "param_configs"
log_dir = "logs_wandb"  # Changed log directory for wandb runs
model_dir = "models_wandb" # Directory to save wandb models

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# --- Custom Callbacks ---
class RewardShapingCallback(BaseCallback):
    """
    Callback for logging dynamic reward metrics to wandb,
    and periodically exporting reward evolution data to files.
    """
    def __init__(self, verbose=0, export_freq=10):
        super().__init__(verbose)
        self.export_freq = export_freq
        self.call_count = 0

    def _on_step(self):
        # Log current reward configuration to wandb on every step
        current_config = metrics_collector.current_reward_config
        for key, value in current_config.items():
            wandb.log({f"reward/{key}": value}, step=self.num_timesteps)
            
        # Export results periodically
        self.call_count += 1
        if self.call_count % self.export_freq == 0:
            export_path = os.path.join(wandb.run.dir, "reward_evolution.json")
            metrics_collector.export_results(export_path)
            # Also save a local copy
            metrics_collector.export_results("reward_evolution_latest.json")
            
        return True


# --- Main Training Function ---
def train():
    # Load base game parameters
    with open(f"{config_dir}/eval.json", "r") as f:
        game_params = json.load(f)

    # Remove the 'rewards' key as it's no longer used by the Env's __init__
    if 'rewards' in game_params:
        del game_params['rewards']

    # Initialize wandb
    run = wandb.init(
        project="snake-rl-llm-rewards",  # New project name for LLM-based rewards
        config={
            # Training Hyperparameters
            "policy_type": "CnnPolicy",  # Using CnnPolicy for image input
            "total_timesteps": 5_000_000,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 2048,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_envs": 32,
            "seed": 42,
            "features_dim": 256,
            # Game Parameters
            **game_params,
            # Initial Reward Configuration
            **metrics_collector.current_reward_config,
            # LLM Shaping Settings
            "llm_call_frequency": 500,  # Call LLM every N episodes
            "llm_model": "gpt-3.5-turbo",  # Or other appropriate model
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    config = wandb.config

    # Define policy kwargs using the feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.features_dim)
    )

    # Create vectorized environment
    vec_env = make_vec_env(
        lambda: SnakeGameEnv(**game_params),
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv
    )

    # Create PPO model
    model = PPO(
        config.policy_type,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cuda' if th.cuda.is_available() else 'cpu',
        tensorboard_log=log_dir,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
    )

    # --- Callbacks ---
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"{model_dir}/{run.id}",
        model_save_freq=config.n_steps * config.n_envs * 5,
        log="all",
        verbose=2,
    )
    
    # Custom callback for reward shaping logging
    reward_callback = RewardShapingCallback(verbose=1, export_freq=50)
    
    # Combine all callbacks
    callbacks = [wandb_callback, reward_callback]

    # --- Training ---
    # Determine number of training iterations based on total_timesteps
    timesteps_per_iteration = 100_000
    iterations = config.total_timesteps // timesteps_per_iteration

    try:
        for i in range(iterations):
            print(f"\n--- Training Iteration {i+1}/{iterations} ---")
            # Log current reward configuration at start of iteration
            print(f"Current reward config: {metrics_collector.current_reward_config}")
            
            model.learn(
                total_timesteps=timesteps_per_iteration,
                progress_bar=True,
                tb_log_name=f"PPO_LLM_Rewards_{run.id}",
                reset_num_timesteps=False,
                callback=callbacks
            )
            
            # Save a checkpoint after each iteration
            model.save(f'{model_dir}/{run.id}/checkpoint_{i+1}')

        # Final save
        model.save(f'{model_dir}/{run.id}/final_model')

    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        model.save(f'{model_dir}/{run.id}/interrupted_model')
    finally:
        # Export final reward evolution data
        export_path = os.path.join(wandb.run.dir, "reward_evolution_final.json")
        metrics_collector.export_results(export_path)
        
        # Close the environment
        vec_env.close()
        
        # Finish the wandb run
        run.finish()
        print("Training finished and run closed.")


if __name__ == "__main__":
    train()