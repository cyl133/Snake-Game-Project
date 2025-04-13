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
LLM_CALL_FREQUENCY = 100000  # Episodes before LLM update
METRICS_LOG_FREQUENCY = 100
N_ENVS = 128  # Number of environments
USE_LLM = True  # SET THIS TO FALSE TO DISABLE LLM COMPLETELY

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def format_llm_prompt(metrics, current_config, history=None, max_history=20):
    if not metrics or metrics.get("episodes_collected", 0) == 0:
        return ""
    
    # Format history with specific changes and results
    history_str = "**History (Recent Updates):**\n"
    if history:
        recent_history = history[-max_history:]
        for i, entry in enumerate(recent_history):
            old_r = entry["old_rewards"]
            new_r = entry["new_rewards"]
            hist_metrics = entry["metrics"]
            
            # Show what changed
            changes = []
            for key in new_r:
                if key in old_r and old_r[key] != new_r[key]:
                    changes.append(f"{key}: {old_r[key]} → {new_r[key]}")
            
            # Include performance before the change
            history_str += f"Update {i+1}:\n"
            history_str += f"- Changes: {', '.join(changes)}\n"
            history_str += f"- Prior performance: {hist_metrics['avg_food_per_episode']:.1f} food, {hist_metrics['avg_episode_length']:.1f} steps\n"
    
    metrics_str = f"""
**Current Performance ({metrics['episodes_collected']} episodes):**
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

    # Add RL principles and reward shaping guidance
    reward_shaping_guide = """
**Reward Shaping Principles:**

1. **Balance & Scale:** Keep rewards proportional. Food reward should generally be 20-100x the step penalty.

2. **Metric Analysis Guidelines:**
   - Low food eaten + short episodes → Increase food_reward, add distance_reduction_reward
   - High wall death % → Increase wall_avoidance_bonus, increase death_penalty
   - High self-collision % → Add loop_penalty, increase exploration_bonus
   - Low efficiency (high steps per food) → Adjust step_penalty, increase distance_reduction_reward
   - Low map coverage → Increase exploration_bonus, decrease wall_follow_penalty

3. **Common Patterns:**
   - If avg_episode_length < 100: Agent dies too quickly; reduce death_penalty, reduce step_penalty
   - If food_eaten < 1.0: Agent isn't finding food; increase food_reward
   - If looping_rate > 30%: Agent is stuck in loops; add loop_penalty
   - If death_wall_pct > 50%: Agent hits walls too often; add wall_avoidance_bonus

4. **Avoid Common Mistakes:**
   - Don't make step_penalty too harsh (-0.05 to -0.5 is reasonable)
   - Don't make death_penalty too extreme (generally -5 to -30)
   - If introducing a new reward component, start small (0.1-1.0)
   - Ensure food_reward (10-50) is significantly higher than any penalty
"""

    prompt = f"""
You are an expert in reinforcement learning reward shaping. Your task is to optimize a reward function for a Snake game agent to maximize food collection and survival time.

{history_str}
{metrics_str}

{rewards_explanation}

{reward_shaping_guide}

**Current Reward Function:**
```json
{json.dumps(current_config, indent=2)}
```

**Task:**
Analyze the metrics and suggest precise reward adjustments. Focus on the RELATIVE PROPORTIONS between rewards rather than absolute values. Keep all rewards within the recommended ranges. Make incremental changes (±10-50% maximum per parameter) rather than 
drastic ones. Consider trade-offs between exploration and exploitation.

For each change you make, consider its effect relative to other rewards. For example, if you increase food_reward, consider whether to adjust step_penalty proportionally.

Provide ONLY the updated JSON configuration.
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
    def __init__(self, check_freq=LLM_CALL_FREQUENCY, log_freq=100, verbose=1, use_llm=USE_LLM, initial_rewards=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_freq = log_freq  # More frequent logging
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
                
                # Log basic metrics more frequently
                if self.episode_count % self.log_freq == 0:
                    metrics = self._aggregate_metrics()
                    if wandb.run:
                        wandb.log({f"metrics/{k}": v for k, v in metrics.items()}, 
                                 step=self.num_timesteps)
                
                # Full metrics collection and potential LLM call less frequently
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
        log_freq=METRICS_LOG_FREQUENCY,
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