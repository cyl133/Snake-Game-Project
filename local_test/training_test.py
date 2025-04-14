import json
import os
import numpy as np
import torch as th
import wandb
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # For type hinting if needed
from wandb.integration.sb3 import WandbCallback
from collections import defaultdict
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import base64
import io
import time # For finding latest model

# Import custom components
from feature_extractor import CustomCNN
from gym_env import SnakeGameEnv
from llm_reward_shaper import LLM_MODEL, LLM_API_URL, GOOGLE_API_KEY, get_reward_for_step, DEFAULT_REWARD_CONFIG

# --- Configuration ---
CONFIG_DIR = "param_configs"
LOG_DIR = "logs_wandb"
MODEL_DIR = "models_wandb" # Directory for saving models via WandbCallback
MODEL_LOAD_DIR = "./models" # Directory to check for loading existing models
LLM_CALL_FREQUENCY = 50000
METRICS_LOG_FREQUENCY = 1024
N_ENVS = 32
USE_LLM = True

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_LOAD_DIR, exist_ok=True) # Ensure load directory exists

def format_llm_prompt(metrics, current_config, history=None, max_history=10):
    if not metrics or metrics.get("episodes_collected", 0) == 0:
        return ""
    
    metrics_str = f"""
**Current Performance ({metrics['episodes_collected']} episodes since last update):**
- Average Episode Length: {metrics['avg_episode_length']:.2f} steps
- Average Food Eaten: {metrics['avg_food_per_episode']:.2f}
- Average Max Snake Length: {metrics['avg_max_snake_length']:.2f}
- Success Rate: {metrics['success_rate_pct']:.2f}% (episodes with ≥1 food)
- Efficiency: {metrics['efficiency_steps_per_food']:.2f} steps per food

**Death Analysis:**
- Wall collisions: {metrics['death_wall_pct']:.2f}%
- Self collisions: {metrics['death_self_pct']:.2f}%
- Timeouts: {metrics['death_timeout_pct']:.2f}%

**Behavior Analysis:**
- Map Coverage: {metrics['avg_map_coverage_pct']:.2f}%
- Looping Behavior: {metrics['looping_rate_pct']:.2f}%
- Average Turns: {metrics['avg_turns_per_episode']:.2f} per episode
- Center Area Visits: {metrics['avg_center_visits']:.2f} per episode
"""

    # Add detailed explanation of how rewards are calculated
    rewards_explanation = """
**How Rewards Are Calculated:**
- `food_reward`: Added when the snake eats food
- `death_penalty`: Applied when the snake dies (collides with wall or itself)
- `step_penalty`: Small penalty applied on every step (encourages efficiency)
- `center_bonus`: Bonus when snake visits the center area of the map
- `loop_penalty`: Penalty when snake shows looping behavior (revisiting same path)
- `wall_follow_penalty`: Penalty when snake stays near walls
- `exploration_bonus`: Reward for visiting new cells (unexplored areas)
- `consecutive_food_bonus`: Additional bonus for eating food in succession (multiplied by count)
- `distance_reduction_reward`: Reward for moving closer to food
- `wall_avoidance_bonus`: Reward for staying away from walls

Positive values encourage behaviors, negative values discourage them.
"""


    prompt = f"""
You are an expert in reinforcement learning reward shaping. Your task is to optimize a reward function for a Snake game agent based on its performance trends and current behavior in order to maximize food eaten and minimize death.

**Metric Trends:**
(Refer to the attached image for recent performance trends across reward updates.)

**Current Metrics (Since Last Update):**
{metrics_str}

**How Rewards Are Calculated:**
{rewards_explanation}

**Current Reward Function:**
```json
{json.dumps(current_config, indent=2)}
```

**Task:**
Analyze the attached image trends, current metrics, and current rewards using the guidelines.
Decide if reward adjustments are needed now to improve performance OR if the agent needs more time to train under the current rewards (e.g., if performance is still clearly improving or highly unstable).

- **If adjustments are needed:** Suggest precise, incremental (±10-50%), proportional reward changes based on the analysis. Keep rewards within recommended ranges.
- **If no changes are needed now:** Indicate this by returning the *current* reward configuration unchanged.

Provide ONLY the JSON reward configuration (either updated or the current one).
"""
    return prompt.strip()

def call_llm(prompt_text, image_data=None):
    """LLM call function supporting text and optional image input."""
    if not GOOGLE_API_KEY:
        print("API key not set. Skipping LLM call.")
        return None

    print("\n--- LLM PROMPT (Text) ---")
    print(prompt_text)
    if image_data:
        print("--- LLM PROMPT (Image Attached) ---")

    headers = {"Content-Type": "application/json"}

    # Construct parts list for the API payload
    parts = [{"text": prompt_text}]
    if image_data:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": image_data
            }
        })

    data = {"contents": [{"parts": parts}]}

    try:
        # Increased timeout for potentially larger payload
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        response_json = response.json()

        # Extract text response (assuming LLM still outputs JSON in text)
        if 'candidates' in response_json and response_json['candidates']:
            candidate_content = response_json['candidates'][0].get('content', {})
            if 'parts' in candidate_content and candidate_content['parts']:
                text = candidate_content['parts'][0].get('text', '')

                # Extract JSON from the text response
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = text[json_start:json_end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"LLM response was not valid JSON: {e}")
                        print(f"Received text: {text}") # Log the invalid response
                        return None
                else:
                     print(f"LLM response did not contain JSON: {text}")
                     return None # No JSON found
            else:
                print("LLM response structure unexpected (no parts).")
                return None
        else:
             print("LLM response structure unexpected (no candidates).")
             return None

    except requests.exceptions.Timeout:
        print("Error calling LLM: Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"LLM Response Status Code: {e.response.status_code}")
             print(f"LLM Response Text: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")

    return None

class RewardUpdateCallback(BaseCallback):
    def __init__(self, check_freq=LLM_CALL_FREQUENCY, log_freq=100, verbose=1, use_llm=USE_LLM, initial_rewards=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_freq = log_freq
        self.episode_count = 0
        self.stats = []
        self.reward_history = []
        self.current_rewards = initial_rewards.copy()
        self.game_params = None
        self.use_llm = use_llm
        print(f"RewardUpdateCallback initialized with LLM {'ENABLED' if use_llm else 'DISABLED'}")
        
    def _plot_metrics_history(self, max_history_points=20):
        """Generates and encodes a plot of key metrics from reward_history."""
        if not self.reward_history:
            print("No history to plot.")
            return None

        history_to_plot = self.reward_history[-max_history_points:]
        if not history_to_plot:
             print("Not enough history points to plot.")
             return None

        steps = [entry['step'] for entry in history_to_plot]

        # Select key metrics to plot from the 'metrics' dict within each history entry
        metrics_to_plot = {
            'Avg Food': [entry['metrics'].get('avg_food_per_episode', 0) for entry in history_to_plot],
            'Avg Length': [entry['metrics'].get('avg_episode_length', 0) for entry in history_to_plot],
            'Success %': [entry['metrics'].get('success_rate_pct', 0) for entry in history_to_plot],
            'Wall Death %': [entry['metrics'].get('death_wall_pct', 0) for entry in history_to_plot],
            'Self Death %': [entry['metrics'].get('death_self_pct', 0) for entry in history_to_plot],
            'Looping %': [entry['metrics'].get('looping_rate_pct', 0) for entry in history_to_plot],
            'Efficiency (Steps/Food)': [entry['metrics'].get('efficiency_steps_per_food', float('inf')) for entry in history_to_plot],
        }

        # Filter out infinite efficiency values for plotting
        efficiency = metrics_to_plot['Efficiency (Steps/Food)']
        finite_efficiency_steps = [s for s, e in zip(steps, efficiency) if np.isfinite(e)]
        finite_efficiency_values = [e for e in efficiency if np.isfinite(e)]

        num_plots = len(metrics_to_plot)
        if num_plots == 0:
            return None

        plt.figure(figsize=(10, 2.5 * num_plots)) # Adjusted figsize

        plot_index = 1
        for key, values in metrics_to_plot.items():
            plt.subplot(num_plots, 1, plot_index)
            if key == 'Efficiency (Steps/Food)':
                 if finite_efficiency_values: # Only plot if there's finite data
                      plt.plot(finite_efficiency_steps, finite_efficiency_values, marker='o', linestyle='-')
                 else:
                      # Optionally plot nothing or a placeholder if no finite values
                      plt.text(0.5, 0.5, 'No Finite Efficiency Data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            else:
                plt.plot(steps, values, marker='o', linestyle='-')

            plt.title(key)
            plt.ylabel("Value")
            if plot_index == num_plots:
                plt.xlabel("Training Timestep")
            else:
                 plt.xticks([]) # Hide x-axis labels for upper plots
            plt.grid(True)
            plot_index += 1

        plt.suptitle("Recent Metrics Evolution (at time of LLM updates)", y=1.0) # Adjusted title and y
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout slightly for suptitle

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close() # Close the plot to free memory
        buf.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        print(f"Generated metrics plot image (base64 encoded) for LLM.")
        return image_base64

    def _on_step(self):
        # Check for episode completions
        for i, done in enumerate(self.locals.get("dones", [])):
            if done and "episode_stats" in self.locals["infos"][i]:
                self.episode_count += 1
                self.stats.append(self.locals["infos"][i]["episode_stats"])
                
                # Log basic metrics more frequently
                if self.episode_count % self.log_freq == 0:
                    metrics = self._aggregate_metrics()
                    if wandb.run:
                        wandb.log({f"metrics/{k}": v for k, v in metrics.items()}, 
                                 step=self.num_timesteps)
                
                # Full metrics collection and potential LLM call less frequently
                if self.episode_count % self.check_freq == 0:
                    print(f"\n--- Episode {self.episode_count}: Collecting Metrics & Calling LLM ---")
                    # Aggregate metrics from the *current* collection period
                    current_metrics = self._aggregate_metrics()
                    if not current_metrics:
                         print("No metrics collected since last update, skipping LLM call.")
                         self.stats = [] # Reset anyway
                         return True

                    # Log current metrics to wandb
                    if wandb.run:
                        wandb.log({f"metrics/{k}": v for k, v in current_metrics.items()},
                                 step=self.num_timesteps)

                    # Only call LLM if enabled
                    if self.use_llm:
                        print("LLM reward updating is ENABLED. Requesting reward update...")

                        # Generate plot from *historical* data and encode it
                        plot_base64 = self._plot_metrics_history() # Uses self.reward_history

                        # Format text prompt using *current* metrics and config
                        # Pass self.reward_history so format_llm_prompt can decide (even if unused)
                        prompt_text = format_llm_prompt(current_metrics, self.current_rewards, self.reward_history)

                        # Call LLM with text and image
                        new_rewards = call_llm(prompt_text, image_data=plot_base64)

                        if new_rewards:
                            print(f"New rewards from LLM: {new_rewards}")

                            # Save history: Use the CURRENT metrics collected just before this call
                            self.reward_history.append({
                                "step": self.num_timesteps,
                                "episode": self.episode_count,
                                "metrics": current_metrics, # Metrics leading to this update
                                "old_rewards": self.current_rewards.copy(),
                                "new_rewards": new_rewards
                            })

                            # Update current rewards
                            self.current_rewards.update(new_rewards)

                            # Log new rewards to wandb
                            if wandb.run:
                                wandb.log({f"rewards/{k}": v for k, v in self.current_rewards.items()},
                                        step=self.num_timesteps)

                            # Update environment rewards
                            self._switch_environment()
                        else:
                            print(f"LLM call failed or returned None at step {self.num_timesteps}. Rewards not updated.")

                    else: # if not self.use_llm:
                        print("LLM reward updating is DISABLED. Using fixed rewards.")
                        # Still log the fixed rewards for consistency
                        if wandb.run:
                            wandb.log({f"rewards/{k}": v for k, v in self.current_rewards.items()},
                                    step=self.num_timesteps)
                    
                    # Reset stats collection *after* potential LLM call and history saving
                    self.stats = []
        
        return True
        
    def _switch_environment(self):
        """Apply reward updates to environments directly"""
        print("Updating rewards in all environments...")
        vec_env = self.model.get_env()
        
        # Loop through each environment to update rewards
        for i in range(len(vec_env.envs)):
            env = vec_env.envs[i].unwrapped
            # Update the reward config in-place
            env.reward_config = self.current_rewards.copy()
        
        print(f"Updated rewards in all environments: {self.current_rewards}")
        
    def _aggregate_metrics(self):
        """Aggregate stats from completed episodes"""
        if not self.stats:
            return {}
            
        metrics = defaultdict(list)
        death_causes = defaultdict(int)
        total_episodes = len(self.stats)
        
        for stat in self.stats:
            # Existing metrics
            metrics["length"].append(stat["length"])
            metrics["food_eaten"].append(stat["food_eaten"])
            metrics["max_length"].append(stat["max_length"])
            metrics["map_coverage"].append(stat.get("map_coverage", 0))
            death_causes[stat.get("death_cause", "unknown")] += 1
            
            # NEW METRICS
            metrics["looping_detected"].append(1 if stat.get("looping", False) else 0)
            metrics["turns"].append(stat.get("turns", 0))
            metrics["center_visits"].append(stat.get("center_visits", 0))
            
            # Calculate efficiency metrics
            if stat["food_eaten"] > 0:
                metrics["steps_per_food"].append(stat["length"] / stat["food_eaten"])
            
            # Track episodes with at least one food
            metrics["success_rate"].append(1 if stat["food_eaten"] > 0 else 0)
        
        # Calculate aggregates
        result = {
            "episodes_collected": total_episodes,
            "avg_episode_length": np.mean(metrics["length"]),
            "avg_food_per_episode": np.mean(metrics["food_eaten"]),
            "avg_max_snake_length": np.mean(metrics["max_length"]),
            "avg_map_coverage_pct": np.mean(metrics["map_coverage"]) * 100 if metrics["map_coverage"] else 0,
            
            # NEW AGGREGATED METRICS
            "avg_turns_per_episode": np.mean(metrics["turns"]) if metrics["turns"] else 0,
            "avg_center_visits": np.mean(metrics["center_visits"]) if metrics["center_visits"] else 0,
            "looping_rate_pct": np.mean(metrics["looping_detected"]) * 100 if metrics["looping_detected"] else 0,
            "success_rate_pct": np.mean(metrics["success_rate"]) * 100 if metrics["success_rate"] else 0,
            "efficiency_steps_per_food": np.mean(metrics["steps_per_food"]) if metrics["steps_per_food"] else float('inf'),
        }
        
        # Calculate death percentages
        result["death_wall_pct"] = death_causes.get("wall", 0) / total_episodes * 100
        result["death_self_pct"] = death_causes.get("self", 0) / total_episodes * 100
        result["death_timeout_pct"] = death_causes.get("timeout", 0) / total_episodes * 100
        
        return result
        
    def _on_training_end(self):
        """Save reward history at the end of training"""
        try:
            with open("reward_evolution.json", "w") as f:
                json.dump(self.reward_history, f, indent=2)
            print("Saved reward evolution history")
        except Exception as e:
            print(f"Error saving history: {e}")


def train():
    # Load game parameters
    with open(f"{CONFIG_DIR}/eval.json", "r") as f:
        game_params = json.load(f)

    # Extract and complete reward config
    reward_config = game_params.get("reward_config")
    if not reward_config:
         reward_config = game_params.get("rewards")
         if not reward_config:
              raise ValueError("No reward configuration found in eval.json under 'reward_config' or 'rewards'")

    expected_keys = list(DEFAULT_REWARD_CONFIG.keys())
    for key in expected_keys:
        if key not in reward_config:
             print(f"Warning: Adding missing key '{key}' to initial reward config with value 0.0")
             reward_config[key] = 0.0

    # Initialize wandb
    run = wandb.init(
        project="snake-rl-simple",
        config={
            "policy_type": "CnnPolicy",
            "total_timesteps": 5_000_0000,
            "n_envs": N_ENVS,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 2048,
            "gae_lambda": 0.99,
            "ent_coef": 0.2, # Log this hyperparameter
            "policy_kwargs_net_arch": "pi=[128, 64], vf=[256, 256, 128]",
            "game_params": game_params,
            "initial_rewards": reward_config,
            "llm_freq": LLM_CALL_FREQUENCY,
            "use_llm": USE_LLM
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    # Define policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[256, 256, 128])
    )

    # Create environment (needed for both loading and creating)
    vec_env = make_vec_env(
        lambda: SnakeGameEnv(**game_params, reward_config=reward_config.copy()),
        n_envs=N_ENVS
    )

    # --- Try to Load Existing Model ---
    load_path: Optional[str] = None
    latest_mtime: float = 0.0

    print(f"Checking for existing models in {MODEL_LOAD_DIR}...")
    if os.path.isdir(MODEL_LOAD_DIR):
        try:
            for filename in os.listdir(MODEL_LOAD_DIR):
                if filename.endswith(".zip"):
                    filepath = os.path.join(MODEL_LOAD_DIR, filename)
                    if os.path.isfile(filepath): # Ensure it's a file
                        mtime = os.path.getmtime(filepath)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            load_path = filepath
        except OSError as e:
            print(f"Warning: Could not read directory {MODEL_LOAD_DIR}: {e}")


    # Define common PPO parameters
    ppo_params = {
        "policy": "CnnPolicy",
        "env": vec_env,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": "cuda" if th.cuda.is_available() else "cpu",
        "tensorboard_log": LOG_DIR,
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 2048,
        "gae_lambda": 0.99,
        "ent_coef": 0.05
    }

    # --- Load or Create Model ---
    if load_path:
        print(f"Loading existing model from: {load_path}")
        try:
            model = PPO.load(
                load_path,
                env=vec_env, # **Crucial:** Set the environment for the loaded model
                device=ppo_params["device"],
                tensorboard_log=LOG_DIR,
                # Pass other params to ensure consistency for continued training,
                # although many are loaded from the zip. Passing env is most important.
                learning_rate=ppo_params["learning_rate"],
                n_steps=ppo_params["n_steps"],
                batch_size=ppo_params["batch_size"],
                gae_lambda=ppo_params["gae_lambda"],
                ent_coef=ppo_params["ent_coef"],
                policy_kwargs=ppo_params["policy_kwargs"]
            )
            print(f"Model loaded successfully. Current timestep: {model.num_timesteps}")
        except Exception as e:
             print(f"Error loading model from {load_path}: {e}")
             print("Creating new model from scratch.")
             model = PPO(**ppo_params)

    else:
        print("No suitable model found in ./models/. Training from scratch.")
        model = PPO(**ppo_params)


    # --- Create Callbacks ---
    # Note: WandbCallback saves to MODEL_DIR (models_wandb), not MODEL_LOAD_DIR (models)
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"{MODEL_DIR}/{run.id}", # Saves under models_wandb/run_id/
        model_save_freq=50_000, # Consider saving more frequently if needed
        log="all"
    )

    reward_callback = RewardUpdateCallback(
        check_freq=LLM_CALL_FREQUENCY,
        log_freq=METRICS_LOG_FREQUENCY,
        use_llm=USE_LLM,
        initial_rewards=reward_config # Pass initial config
    )
    callbacks = [wandb_callback, reward_callback]

    # --- Train ---
    try:
        # Determine total timesteps remaining if loading a model
        total_timesteps_config = run.config.get("total_timesteps", 5_000_000)
        timesteps_to_train = total_timesteps_config - model.num_timesteps
        if timesteps_to_train <= 0:
             print("Model already trained for total timesteps. Exiting.")
        else:
             print(f"Training for an additional {timesteps_to_train} timesteps...")
             model.learn(
                 total_timesteps=timesteps_to_train, # Train for remaining steps
                 callback=callbacks,
                 progress_bar=True,
                 tb_log_name=f"PPO_Snake_{run.id}",
                 reset_num_timesteps=False # IMPORTANT: Do not reset timesteps when continuing training
             )
             # Save final model explicitly in MODEL_LOAD_DIR if desired
             final_save_path = os.path.join(MODEL_LOAD_DIR, f"final_model_{run.id}.zip")
             model.save(final_save_path)
             print(f"Final model saved to {final_save_path}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        # Save interrupted model to both Wandb dir and local dir
        interrupted_wandb_path = f"{MODEL_DIR}/{run.id}/interrupted_model"
        model.save(interrupted_wandb_path)
        interrupted_local_path = os.path.join(MODEL_LOAD_DIR, f"interrupted_model_{run.id}.zip")
        model.save(interrupted_local_path)
        print(f"Interrupted model saved to {interrupted_wandb_path} and {interrupted_local_path}")
    finally:
        if 'vec_env' in locals() and vec_env is not None:
            vec_env.close()
        if 'run' in locals() and run is not None:
            run.finish()
        print("Training finished or interrupted.")


if __name__ == "__main__":
    train()