import json
import os
import time
import numpy as np
import torch as th
import wandb
import requests # Needed for LLM call
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # Import DummyVecEnv for single process debugging
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from collections import defaultdict
from typing import Dict, Optional, List

# Import custom components
from feature_extractor import CustomCNN
from gym_env import SnakeGameEnv
# Import the global config holder and LLM details
from llm_reward_shaper import metrics_collector, GOOGLE_API_KEY, LLM_API_URL, LLM_MODEL

# --- Configuration ---
config_dir = "param_configs"
log_dir = "logs_wandb"
model_dir = "models_wandb"
LLM_CALL_FREQUENCY_EPISODES = 10000 # How many *total* episodes across all envs trigger LLM
# Set USE_SUBPROC_VEC_ENV to False for easier debugging of callbacks/LLM interaction
USE_SUBPROC_VEC_ENV = True # Set to False to run in single process

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# --- LLM Interaction Logic (moved here from MetricsCollector) ---

def format_llm_prompt(
    aggregated_metrics: dict,
    current_config: dict,
    history: List[dict],
    max_history: int = 10
) -> str:
    """Formats the prompt for the Gemini API based on aggregated metrics and recent history."""
    if not aggregated_metrics or aggregated_metrics.get("episodes_collected", 0) == 0:
        return ""

    # --- Format History Section ---
    history_str = "**History (Recent Updates):**\n"
    if not history:
        history_str += "No history available yet.\n"
    else:
        # Get the last N entries (or fewer if history is short)
        recent_history = history[-max_history:]
        for i, entry in enumerate(recent_history):
            iter_num = entry.get("llm_iteration", i + 1) # Use LLM iteration number
            step_num = entry.get("global_step", "N/A")
            metrics = entry.get("metrics_used", {})
            config_after = entry.get("config_after", entry.get("config_before", {})) # Show config resulting from this iteration

            # Select key metrics and config values for brevity
            hist_metrics_summary = (
                f"Avg Food: {metrics.get('avg_food_per_episode', 0):.2f}, "
                f"Avg Len: {metrics.get('avg_episode_length', 0):.2f}, "
                f"Coverage: {metrics.get('avg_map_coverage_pct', 0):.1f}%, "
                f"Looping: {metrics.get('looping_rate_pct', 0):.1f}%"
            )
            # Select key config values
            hist_config_summary = (
                 f"food={config_after.get('food_reward', 0):.2f}, "
                 f"death={config_after.get('death_penalty', 0):.2f}, "
                 f"step={config_after.get('step_penalty', 0):.3f}, "
                 f"expl={config_after.get('exploration_bonus', 0):.2f}, "
                 f"loop={config_after.get('loop_penalty', 0):.2f}"
            )

            history_str += (
                f"Update {iter_num} (Step {step_num}):\n"
                f"- Metrics -> {hist_metrics_summary}\n"
                f"- Resulting Config -> {{{hist_config_summary}}}\n"
            )
        history_str += "\n" # Add blank line after history

    # --- Format Current Metrics Section ---
    current_metrics_str = f"""
**Current Performance (Since Last Update - {aggregated_metrics['episodes_collected']} episodes):**
- Average Episode Length: {aggregated_metrics['avg_episode_length']:.2f} steps
- Average Food Eaten: {aggregated_metrics['avg_food_per_episode']:.2f}
- Average Steps per Food: {aggregated_metrics['avg_steps_per_food']:.2f}
- Average Max Snake Length: {aggregated_metrics['avg_max_snake_length']:.2f}
- Death Causes: Wall ({aggregated_metrics['death_wall_pct']:.2f}%), Self-Collision ({aggregated_metrics['death_self_pct']:.2f}%), Timeout/Won ({aggregated_metrics['death_other_pct']:.2f}%)
- Average Map Coverage: {aggregated_metrics['avg_map_coverage_pct']:.2f}%
- Average Center Visits per Episode: {aggregated_metrics['avg_center_visits']:.2f}
- Looping Episode Rate: {aggregated_metrics['looping_rate_pct']:.2f}%
- Average Action Entropy: {aggregated_metrics['avg_action_entropy']:.3f} (Higher means more random actions)
- Average Turns per Episode: {aggregated_metrics['avg_turns_per_episode']:.2f}
"""

    # --- Assemble Final Prompt ---
    prompt = f"""
You are an expert in Snake RL reward shaping. Analyze the behavior of a Snake RL agent. 

{history_str}
{current_metrics_str}

**Current Reward Function Before This Update:**
```json
{json.dumps(current_config, indent=2)}
```

**Task:**
Based on the **current performance metrics** and potentially informed by the **recent history**, suggest modifications to the reward function JSON below to encourage better performance (eat as much food as possible without dying).

**Guidelines:**
- Be very aggressive and very purposeful in your reward updates.
- Provide ONLY the updated JSON configuration. Do NOT include explanations or surrounding text.

**Provide the updated reward JSON here:**
```json
{{
  "food_reward": {current_config.get('food_reward', 1.0)},
  "death_penalty": {current_config.get('death_penalty', -1.0)},
  "step_penalty": {current_config.get('step_penalty', -0.01)},
  "center_bonus": {current_config.get('center_bonus', 0.0)},
  "loop_penalty": {current_config.get('loop_penalty', 0.0)},
  "wall_follow_penalty": {current_config.get('wall_follow_penalty', 0.0)},
  "exploration_bonus": {current_config.get('exploration_bonus', 0.0)}
}}
```
"""
    return prompt.strip()

def call_gemini_api(prompt: str) -> Optional[dict]:
    """Calls the Gemini API and returns the parsed new reward config or None."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("[Callback] Google API key not set. Skipping LLM call.")
        print("\n--- LLM PROMPT (SKIPPED) ---")
        print(prompt)
        print("---------------------------\n")
        return None

    print("\n--- [Callback] LLM PROMPT ---")
    print(prompt)
    print("-----------------------------\n")

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    new_config = None
    try:
        print(f"[Callback] Sending request to Gemini API: {LLM_API_URL}")
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=60) # Add timeout

        print(f"\n--- [Callback] LLM RAW RESPONSE (Status: {response.status_code}) ---")
        try:
            response_json = response.json()
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("Could not decode JSON response:")
            print(response.text)
        print("------------------------------------\n")

        response.raise_for_status()

        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                llm_response_text = candidate['content']['parts'][0]['text']
                print(f"[Callback] LLM Extracted Response Text:\n{llm_response_text}\n")
                # Parse JSON from the extracted text
                json_match = llm_response_text.strip()
                if json_match.startswith("```json"):
                    json_match = json_match[len("```json"):].strip()
                if json_match.startswith("```"):
                    json_match = json_match[len("```"):].strip()
                if json_match.endswith("```"):
                    json_match = json_match[:-len("```")].strip()

                if json_match.startswith("{") and json_match.endswith("}"):
                     new_config = json.loads(json_match)
                else:
                     print("[Callback] Warning: LLM response JSON structure not found.")
                     # Try fallback parsing
                     json_start = llm_response_text.find('{')
                     json_end = llm_response_text.rfind('}') + 1
                     if json_start != -1 and json_end != 0:
                          try:
                              new_config = json.loads(llm_response_text[json_start:json_end])
                          except json.JSONDecodeError:
                               print("[Callback] Fallback JSON parsing also failed.")
                     else:
                          print("[Callback] Could not extract JSON from LLM response.")

            else:
                 print("[Callback] Warning: Unexpected response structure from Gemini.")
        else:
             print("[Callback] Warning: No candidates found in Gemini response.")
             if 'promptFeedback' in response_json and 'blockReason' in response_json['promptFeedback']:
                   print(f"Prompt blocked. Reason: {response_json['promptFeedback']['blockReason']}")

    except requests.exceptions.Timeout:
         print("[Callback] Error: Request to Gemini API timed out.")
    except requests.exceptions.RequestException as e:
        print(f"[Callback] Error calling Gemini API: {e}")
        if e.response is not None:
            pass # Already printed raw response
    except json.JSONDecodeError as e:
         print(f"[Callback] Failed to parse JSON from LLM response: {e}")
         print(f"LLM Raw Response was: {llm_response_text if 'llm_response_text' in locals() else 'N/A'}")
    except Exception as e:
        print(f"[Callback] An unexpected error occurred during LLM call: {e}")

    return new_config


# --- Custom Callbacks ---
class LLMTriggerCallback(BaseCallback):
    """
    Callback to:
    1. Track total completed episodes across all environments.
    2. Collect stats from completed episodes.
    3. Periodically trigger an LLM call based on aggregated stats.
    4. Update the global reward configuration.
    5. Log aggregated metrics and reward config changes to WandB.
    """
    def __init__(self, check_freq_episodes: int, verbose=1):
        super().__init__(verbose)
        self.check_freq_episodes = check_freq_episodes
        self.total_episodes_completed = 0
        # Store metrics from episodes completed since the last LLM call
        self.collected_episode_stats = []
        # Store history of configs and metrics for saving
        self.reward_config_history = []
        self.llm_call_count = 0
        print(f"[Callback] Initialized. LLM check frequency: {check_freq_episodes} total episodes.")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For Higgins, the environment metadata is stored in the local variables.
        """
        # Check for finished episodes in parallel environments
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Episode finished in environment i
                info = self.locals["infos"][i]
                # Check if our final stats are present (from gym_env.py)
                if "episode_stats" in info:
                    self.total_episodes_completed += 1
                    self.collected_episode_stats.append(info["episode_stats"])

                    if self.verbose > 1:
                         print(f"[Callback] Env {i} finished episode {self.total_episodes_completed}. Stats: {info['episode_stats']}")

                    # Check if it's time to trigger the LLM update
                    if self.total_episodes_completed > 0 and \
                       self.total_episodes_completed % self.check_freq_episodes == 0:
                        if self.verbose > 0:
                             print(f"\n--- [Callback] Total Episodes {self.total_episodes_completed}: Triggering LLM Reward Update ---")
                        self._trigger_llm_update()
                        # Clear collected stats after triggering
                        self.collected_episode_stats = []

        # Log current reward config periodically (e.g., every N steps) for denser tracking
        log_freq_steps = 1000 # Log reward config every 1000 global steps
        if self.num_timesteps % log_freq_steps == 0:
             current_config = metrics_collector.current_reward_config
             log_dict = {f"reward_config/{k}": v for k, v in current_config.items()}
             wandb.log(log_dict, step=self.num_timesteps)

        return True # Continue training

    def _aggregate_metrics(self) -> dict:
        """Aggregates metrics from the collected episode stats."""
        num_episodes = len(self.collected_episode_stats)
        if num_episodes == 0:
            return {}

        agg = defaultdict(list)
        death_causes = defaultdict(int)
        looping_count = 0

        for stats in self.collected_episode_stats:
            agg["length"].append(stats["length"])
            agg["food_eaten"].append(stats["food_eaten"])
            agg["max_length"].append(stats["max_length"])
            agg["map_coverage"].append(stats["map_coverage"])
            agg["center_visits"].append(stats["center_visits"])
            agg["turns"].append(stats["turns"])
            agg["action_entropy"].append(stats["action_entropy"])
            # Calculate steps per food, handle division by zero
            agg["steps_per_food"].append(stats["length"] / max(1, stats["food_eaten"]))
            death_causes[stats["death_cause"]] += 1
            if stats.get("looping", False): # Check if looping key exists
                 looping_count += 1

        # Calculate averages and percentages
        summary = {}
        summary["episodes_collected"] = num_episodes
        summary["avg_episode_length"] = np.mean(agg["length"])
        summary["avg_food_per_episode"] = np.mean(agg["food_eaten"])
        valid_spf = [s for s in agg["steps_per_food"] if np.isfinite(s)]
        summary["avg_steps_per_food"] = np.mean(valid_spf) if valid_spf else 0
        summary["avg_max_snake_length"] = np.mean(agg["max_length"])
        summary["avg_map_coverage_pct"] = np.mean(agg["map_coverage"]) * 100
        summary["avg_center_visits"] = np.mean(agg["center_visits"])
        summary["looping_rate_pct"] = (looping_count / num_episodes) * 100
        summary["avg_action_entropy"] = np.mean(agg["action_entropy"])
        summary["avg_turns_per_episode"] = np.mean(agg["turns"])

        total_deaths = sum(death_causes.values())
        summary["death_wall_pct"] = death_causes.get("wall", 0) / total_deaths * 100 if total_deaths > 0 else 0
        summary["death_self_pct"] = death_causes.get("self", 0) / total_deaths * 100 if total_deaths > 0 else 0
        # Group timeout/won/other
        other_deaths = total_deaths - death_causes.get("wall", 0) - death_causes.get("self", 0)
        summary["death_other_pct"] = other_deaths / total_deaths * 100 if total_deaths > 0 else 0

        return summary

    def _trigger_llm_update(self):
        """Aggregates metrics, calls LLM, updates global config, and logs."""
        self.llm_call_count += 1
        aggregated_metrics = self._aggregate_metrics()

        if not aggregated_metrics:
            if self.verbose > 0:
                print("[Callback] No episode stats collected since last call. Skipping LLM trigger.")
            return

        # Get current config from the first environment (all should be the same)
        base_env = self.model.get_env().envs[0].unwrapped
        current_config = base_env.reward_config.copy()

        # Log aggregated metrics to WandB
        wandb.log({f"llm_metrics/{k}": v for k, v in aggregated_metrics.items()}, step=self.num_timesteps)
        wandb.log({"llm_call_iteration": self.llm_call_count}, step=self.num_timesteps)


        # Record history before calling LLM
        # IMPORTANT: Clone aggregated_metrics, otherwise it might get modified if reused
        history_entry = {
            "config_before": current_config,
            "metrics_used": aggregated_metrics.copy(),
            "llm_iteration": self.llm_call_count,
            "global_step": self.num_timesteps,
            "total_episodes": self.total_episodes_completed
        }

        # Format prompt and call LLM, passing the history
        prompt = format_llm_prompt(aggregated_metrics, current_config, self.reward_config_history) # Pass history here
        if not prompt:
            print("[Callback] Could not format LLM prompt (no metrics?).")
            # Still append history entry even if prompt fails, to record the metrics state
            self.reward_config_history.append(history_entry)
            return

        new_config_suggestion = call_gemini_api(prompt)

        # Update the reward config in all environments
        if new_config_suggestion and isinstance(new_config_suggestion, dict):
            vec_env = self.model.get_env()
            for env_idx in range(len(vec_env.envs)):
                # Use remotes to update config in subprocesses
                if hasattr(vec_env, 'remotes'):
                    vec_env.remotes[env_idx].send(('set_reward_config', new_config_suggestion))
                    response = vec_env.remotes[env_idx].recv()
                else:
                    # Direct update for DummyVecEnv
                    vec_env.envs[env_idx].unwrapped.reward_config.update(new_config_suggestion)
        else:
             history_entry["config_after"] = current_config # Config didn't change
             print("[Callback] LLM did not return a valid config update.")

        self.reward_config_history.append(history_entry) # Append the completed history entry

        # Periodically save the history to a file
        save_freq_llm_calls = 5 # Save history every 5 LLM calls
        if self.llm_call_count % save_freq_llm_calls == 0:
             self._save_history()

    def _save_history(self, final=False):
         """Saves the reward config history to JSON."""
         suffix = "final" if final else f"iter_{self.llm_call_count}"
         filename_local = f"reward_evolution_{suffix}.json"
         filename_wandb = os.path.join(wandb.run.dir, filename_local) if wandb.run else filename_local

         try:
              with open(filename_local, 'w') as f:
                   json.dump(self.reward_config_history, f, indent=2)
              if self.verbose > 0:
                  print(f"[Callback] Reward evolution data saved to {filename_local}")
              # Save to wandb files if run exists
              if wandb.run:
                   wandb.save(filename_local, base_path=wandb.run.dir, policy="now") # Ensure it uploads
         except Exception as e:
              print(f"[Callback] Error saving reward evolution data: {e}")

    def _on_training_end(self):
        """Called at the end of training."""
        if self.verbose > 0:
            print("[Callback] Training ended. Saving final reward evolution history.")
        self._save_history(final=True)


# --- Main Training Function ---
def train():
    with open(f"{config_dir}/eval.json", "r") as f:
        game_params = json.load(f)
    if 'rewards' in game_params:
        del game_params['rewards'] # Remove old static rewards

    run = wandb.init(
        project="snake-rl-llm-rewards-v2", # Updated project name
        config={
            # Training Hyperparameters
            "policy_type": "CnnPolicy",
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
            "n_envs": 32 if USE_SUBPROC_VEC_ENV else 1, # Adjust n_envs based on VecEnv type
            "seed": 42,
            "features_dim": 256,
            # Game Parameters
            **game_params,
            # Initial Reward Configuration (from global collector)
            **metrics_collector.current_reward_config,
            # LLM Shaping Settings
            "llm_call_frequency_episodes": LLM_CALL_FREQUENCY_EPISODES,
            "llm_model": LLM_MODEL,
            "vec_env_type": "SubprocVecEnv" if USE_SUBPROC_VEC_ENV else "DummyVecEnv"
        },
        sync_tensorboard=True,
        monitor_gym=True, # Important for SB3 to log ep_len_mean etc.
        save_code=True,
    )
    config = wandb.config # Use wandb config

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.features_dim)
    )

    # Choose VecEnv type
    vec_env_cls = SubprocVecEnv if USE_SUBPROC_VEC_ENV else DummyVecEnv
    print(f"Using VecEnv type: {vec_env_cls.__name__}")

    vec_env = make_vec_env(
        lambda: SnakeGameEnv(**game_params),
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=vec_env_cls
        # monitor_dir=log_dir # SB3 Monitor logs go here
    )

    model = PPO(
        config.policy_type,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cuda' if th.cuda.is_available() else 'cpu',
        tensorboard_log=log_dir, # SB3 logs standard metrics here
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
    # WandbCallback syncs SB3 logs and logs gradients/model checkpoints
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"{model_dir}/{run.id}",
        model_save_freq=max(1, config.n_steps * config.n_envs * 10), # Save every 10 rollouts approx
        log="all",
        verbose=2,
    )
    # Our custom callback handles LLM triggering and logging
    llm_callback = LLMTriggerCallback(check_freq_episodes=config.llm_call_frequency_episodes, verbose=1)

    callbacks = [wandb_callback, llm_callback]

    # --- Training ---
    timesteps_per_iteration = 100_000
    iterations = config.total_timesteps // timesteps_per_iteration

    try:
        model.learn(
             total_timesteps=config.total_timesteps,
             progress_bar=True,
             tb_log_name=f"PPO_LLM_Rewards_{run.id}",
             reset_num_timesteps=True, # Start fresh for this run
             callback=callbacks
        )
        # Final save after learn completes
        model.save(f'{model_dir}/{run.id}/final_model')

    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
        model.save(f'{model_dir}/{run.id}/interrupted_model')
    finally:
        # Ensure final history is saved by callback's _on_training_end
        print("Closing environment and finishing run...")
        vec_env.close()
        run.finish()
        print("Training finished and run closed.")


if __name__ == "__main__":
    train()