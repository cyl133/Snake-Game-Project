import json
import os
import numpy as np
import torch as th
import wandb
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from collections import defaultdict
from typing import Dict, List, Optional

# Import custom components
from feature_extractor import CustomCNN
from gym_env import SnakeGameEnv
from llm_reward_shaper import LLM_MODEL, LLM_API_URL, GOOGLE_API_KEY, get_reward_for_step

# --- Configuration ---
CONFIG_DIR = "param_configs"
LOG_DIR = "logs_wandb"
MODEL_DIR = "models_wandb"
LLM_CALL_FREQUENCY = 10000  # Episodes before LLM update
N_ENVS = 128  # Number of environments
USE_LLM = True  # SET THIS TO FALSE TO DISABLE LLM COMPLETELY

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def format_llm_prompt(metrics, current_config, history=None, max_history=10):
    # Same prompt formatting as before
    if not metrics or metrics.get("episodes_collected", 0) == 0:
        return ""
    
    # Format history and metrics sections - simplified for brevity
    history_str = "**History (Recent Updates):**\n"
    if history:
        recent_history = history[-max_history:]
        for i, entry in enumerate(recent_history):
            history_str += f"Update {i+1}: Modified rewards based on performance\n"
    
    metrics_str = f"""
**Current Performance ({metrics['episodes_collected']} episodes):**
- Average Episode Length: {metrics['avg_episode_length']:.2f} steps
- Average Food Eaten: {metrics['avg_food_per_episode']:.2f}
- Average Max Snake Length: {metrics['avg_max_snake_length']:.2f}
- Death Causes: Wall ({metrics['death_wall_pct']:.2f}%), Self ({metrics['death_self_pct']:.2f}%)
- Map Coverage: {metrics['avg_map_coverage_pct']:.2f}%
"""

    prompt = f"""
You are an expert in Snake RL reward shaping. Analyze the behavior of a Snake RL agent.

{history_str}
{metrics_str}

**Current Reward Function:**
```json
{json.dumps(current_config, indent=2)}
```

**Task:**
Based on the current performance metrics, suggest modifications to the reward function JSON.
Provide ONLY the updated JSON configuration.

```json
{{
  "food_reward": {current_config.get('food_reward', 1.0)},
  "death_penalty": {current_config.get('death_penalty', 0)},
  "step_penalty": {current_config.get('step_penalty', 0)},
  "center_bonus": {current_config.get('center_bonus', 0.0)},
  "loop_penalty": {current_config.get('loop_penalty', 0.0)},
  "wall_follow_penalty": {current_config.get('wall_follow_penalty', 0.0)},
  "exploration_bonus": {current_config.get('exploration_bonus', 0.0)}
}}
```
"""
    return prompt.strip()

def call_llm(prompt):
    """Simple LLM call function that returns parsed JSON or None"""
    if not GOOGLE_API_KEY:
        print("API key not set. Skipping LLM call.")
        return None
        
    print("\n--- LLM PROMPT ---")
    print(prompt)
    
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        
        # Extract text from response
        if 'candidates' in response_json and response_json['candidates']:
            text = response_json['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
    except Exception as e:
        print(f"Error calling LLM: {e}")
    
    return None

class RewardUpdateCallback(BaseCallback):
    def __init__(self, check_freq=LLM_CALL_FREQUENCY, verbose=1, use_llm=USE_LLM, initial_rewards=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.stats = []
        self.reward_history = []
        self.current_rewards = initial_rewards.copy()
        self.game_params = None
        self.use_llm = use_llm
        print(f"RewardUpdateCallback initialized with LLM {'ENABLED' if use_llm else 'DISABLED'}")
        
    def _on_step(self):
        # Check for episode completions
        for i, done in enumerate(self.locals.get("dones", [])):
            if done and "episode_stats" in self.locals["infos"][i]:
                self.episode_count += 1
                self.stats.append(self.locals["infos"][i]["episode_stats"])
                
                # Check if it's time for metrics aggregation (with or without LLM)
                if self.episode_count % self.check_freq == 0:
                    print(f"\n--- Episode {self.episode_count}: Collecting Metrics ---")
                    metrics = self._aggregate_metrics()
                    
                    # Log metrics to wandb
                    if wandb.run:
                        wandb.log({f"metrics/{k}": v for k, v in metrics.items()}, 
                                 step=self.num_timesteps)
                    
                    # Only call LLM if enabled
                    if self.use_llm:
                        print("LLM reward updating is ENABLED. Requesting reward update...")
                        prompt = format_llm_prompt(metrics, self.current_rewards, self.reward_history)
                        new_rewards = call_llm(prompt)
                        
                        if new_rewards:
                            print(f"New rewards from LLM: {new_rewards}")
                            
                            # Save history
                            self.reward_history.append({
                                "step": self.num_timesteps,
                                "episode": self.episode_count,
                                "metrics": metrics,
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
                        print("LLM reward updating is DISABLED. Using fixed rewards.")
                        # Still log the fixed rewards for consistency
                        if wandb.run:
                            wandb.log({f"rewards/{k}": v for k, v in self.current_rewards.items()},
                                    step=self.num_timesteps)
                    
                    # Reset stats collection
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
        
        for stat in self.stats:
            metrics["length"].append(stat["length"])
            metrics["food_eaten"].append(stat["food_eaten"])
            metrics["max_length"].append(stat["max_length"])
            metrics["map_coverage"].append(stat["map_coverage"])
            death_causes[stat["death_cause"]] += 1
        
        # Calculate aggregates
        result = {
            "episodes_collected": len(self.stats),
            "avg_episode_length": np.mean(metrics["length"]),
            "avg_food_per_episode": np.mean(metrics["food_eaten"]),
            "avg_max_snake_length": np.mean(metrics["max_length"]),
            "avg_map_coverage_pct": np.mean(metrics["map_coverage"]) * 100
        }
        
        # Calculate death percentages
        total = len(self.stats)
        result["death_wall_pct"] = death_causes.get("wall", 0) / total * 100
        result["death_self_pct"] = death_causes.get("self", 0) / total * 100
        result["death_other_pct"] = (total - death_causes.get("wall", 0) - death_causes.get("self", 0)) / total * 100
        
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
    
    # Extract reward config from eval.json
    reward_config = game_params.get("rewards")
    if not reward_config:
        raise ValueError("No reward configuration found in eval.json")
    
    # Initialize wandb with LLM flag
    run = wandb.init(
        project="snake-rl-simple", 
        config={
            "policy_type": "CnnPolicy",
            "total_timesteps": 5_000_000,
            "n_envs": N_ENVS,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 2048,
            "game_params": game_params,
            "initial_rewards": reward_config,
            "llm_freq": LLM_CALL_FREQUENCY,
            "use_llm": USE_LLM
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    
    # Create policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256)
    )
    
    # Create environment with rewards from eval.json
    vec_env = make_vec_env(
        lambda: SnakeGameEnv(**game_params, reward_config=reward_config.copy()),
        n_envs=N_ENVS
    )
    
    # Create model
    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda" if th.cuda.is_available() else "cpu",
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=2048
    )
    
    # Create callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"{MODEL_DIR}/{run.id}",
        model_save_freq=50_000,
        log="all"
    )
    
    reward_callback = RewardUpdateCallback(
        check_freq=LLM_CALL_FREQUENCY, 
        use_llm=USE_LLM,
        initial_rewards=reward_config
    )
    callbacks = [wandb_callback, reward_callback]
    
    # Train
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"PPO_Snake_{run.id}"
        )
        model.save(f"{MODEL_DIR}/{run.id}/final_model")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(f"{MODEL_DIR}/{run.id}/interrupted_model")
    finally:
        vec_env.close()
        run.finish()
        print("Training finished")


if __name__ == "__main__":
    train()