import json
import os
# ssh e0694448@xlogin.comp.nus.edu 
import torch as th
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

# Assuming gym_env.py and feature_extractor.py are in the same directory or accessible
from feature_extractor import CustomCNN
from gym_env import SnakeGameEnv

# --- Configuration ---
config_dir = "param_configs"
log_dir = "logs_wandb"  # Changed log directory for wandb runs
model_dir = "models_wandb" # Directory to save wandb models

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


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
        project="snake-rl-project",  # Choose your project name
        config={
            # Training Hyperparameters
            "policy_type": "MultiInputPolicy", # Assuming Dict observation space
            "total_timesteps": 5_000_000,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 2048, # Adjusted based on original script: n_envs * n_steps = 32 * 128 = 4096? Let's keep 2048 for now.
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
            # Game Parameters (logged from loaded config, rewards removed)
            **game_params, # Pass the modified game_params here for logging
        },
        sync_tensorboard=True,  # Syncs tensorboard logs
        monitor_gym=True,       # Automatically log gym environments
        save_code=True,         # Saves the main script to wandb
    )

    config = wandb.config # Use wandb config for hyperparameters

    # Define policy kwargs using the feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.features_dim)
    )

    # Create vectorized environment using SubprocVecEnv for parallelism
    # The lambda now uses the game_params dictionary *without* the 'rewards' key
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
        gradient_save_freq=10_000, # Save gradients every 10k steps
        model_save_path=f"{model_dir}/{run.id}", # Save model checkpoints associated with the run
        model_save_freq=config.n_steps * config.n_envs * 5, # Save model every 5 rollouts
        log="all", # Log gradients, parameters, and environment stats
        verbose=2,
    )
    # Add other callbacks if needed, e.g., CheckpointCallback, EvalCallback
    # callbacks = [wandb_callback, other_callback]
    callbacks = [wandb_callback]

    # --- Training ---
    # Determine number of training iterations based on total_timesteps
    # This loop structure allows for potential actions between training phases
    timesteps_per_iteration = 100_000 # Example: learn in chunks
    iterations = config.total_timesteps // timesteps_per_iteration

    try:
        for i in range(iterations):
            print(f"\n--- Training Iteration {i+1}/{iterations} ---")
            model.learn(
                total_timesteps=timesteps_per_iteration,
                progress_bar=True,
                tb_log_name=f"PPO_{run.id}", # Log under a run-specific name
                reset_num_timesteps=False, # Continue timestep count across .learn() calls
                callback=callbacks
            )
            # Optional: Save model manually at the end of each major iteration if needed
            # model.save(f'{model_dir}/{run.id}/manual_checkpoint_{i+1}')

        # Final save after all iterations
        model.save(f'{model_dir}/{run.id}/final_model')

    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        model.save(f'{model_dir}/{run.id}/interrupted_model')
    finally:
        # Close the environment
        vec_env.close()
        # Finish the wandb run
        run.finish()
        print("Training finished and run closed.")


if __name__ == "__main__":
    train()