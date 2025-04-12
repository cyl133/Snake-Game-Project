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
from typing import Dict, Optional, List, Tuple # Add Tuple

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

# Define tunable parameters - wall_layout is handled differently in prompt
TUNABLE_ENV_PARAMS = {
    "gs": {"type": "int", "range": [8, 15], "description": "Grid Size (width/height)"},
    "num_fruits": {"type": "int", "range": [1, 5], "description": "Number of fruits on map"},
    "init_hp": {"type": "int", "range": [50, 500], "description": "Initial snake health points"},
    "max_steps": {"type": "int", "range": [500, 2000], "description": "Max steps per episode before timeout"}
    # Wall layout is generated, not clamped to a range here
}
MAX_WALLS_SUGGESTION = 15 # Limit how many walls LLM suggests

# --- LLM Interaction Logic ---
def format_llm_prompt(
    aggregated_metrics: dict,
    current_reward_config: dict,
    current_env_params: dict, # Contains current gs, num_fruits etc.
    current_wall_layout: Optional[List[Tuple[int, int]]], # Add current walls
    history: List[dict],
    max_history: int = 3 # Keep history even shorter for map complexity
) -> str:
    """Formats prompt asking for rewards, NEXT STAGE env params, and NEXT STAGE wall layout."""
    if not aggregated_metrics or aggregated_metrics.get("episodes_collected", 0) == 0: return ""

    # --- Format History Section (Condensed) ---
    history_str = "**History (Recent Stages):**\n"
    if not history: history_str += "No history available yet.\n"
    else:
        recent_history = history[-max_history:]
        for i, entry in enumerate(recent_history):
             iter_num = entry.get("llm_iteration", i + 1)
             metrics = entry.get("metrics_used", {})
             config_after = entry.get("config_after", entry.get("config_before", {}))
             env_params_before = entry.get("env_params_before", {})
             # Also show number of walls used in that stage
             num_walls_before = len(env_params_before.get("wall_layout", []))

             hist_metrics_summary = f"Avg Food: {metrics.get('avg_food_per_episode', 0):.2f}, Avg Len: {metrics.get('avg_episode_length', 0):.2f}"
             hist_config_summary = f"food={config_after.get('food_reward', 0):.2f}, death={config_after.get('death_penalty', 0):.2f}"
             hist_env_summary = f"gs={env_params_before.get('gs', '?')}, fruits={env_params_before.get('num_fruits', '?')}, walls={num_walls_before}"

             history_str += f"Update {iter_num} (Env: {hist_env_summary}): Metrics -> {hist_metrics_summary}, Resulting Reward -> {{{hist_config_summary}}}\n"
        history_str += "\n"

    # --- Format Current Metrics Section (as before) ---
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
- Average Action Entropy: {aggregated_metrics['avg_action_entropy']:.3f}
- Average Turns per Episode: {aggregated_metrics['avg_turns_per_episode']:.2f}
"""

    # --- Format Current Environment Params ---
    current_env_params_str = "**Current Environment Parameters:**\n```json\n"
    # Only show tunable scalar params here, wall layout below
    current_env_params_str += json.dumps({k: current_env_params.get(k) for k in TUNABLE_ENV_PARAMS}, indent=2)
    current_env_params_str += "\n```\n"

    # --- Format Current Wall Layout ---
    current_wall_str = "**Current Wall Layout:**\n"
    if current_wall_layout:
         current_wall_str += f"```json\n{json.dumps(current_wall_layout)}\n```\n"
    else:
         current_wall_str += "None (Empty Grid)\n"

    # --- Format Tunable Param Ranges (scalar only) ---
    tunable_ranges_str = "**Tunable Environment Parameter Ranges (for next stage scalar suggestions):**\n"
    for key, details in TUNABLE_ENV_PARAMS.items():
         tunable_ranges_str += f"- `{key}` ({details['description']}): {details['range']}\n"

    # --- Assemble Final Prompt ---
    prompt = f"""
You are an expert Snake RL agent trainer designing rewards and curriculum.

{history_str}
{current_metrics_str}
{current_env_params_str}
{current_wall_str}
**Current Reward Function:**
```json
{json.dumps(current_reward_config, indent=2)}
```

**Task:**
Based on the **current performance metrics** and potentially informed by the **recent history**, suggest:
1. Modifications to the **reward function** for the current environment setup.
2. Scalar environment parameters (`gs`, `num_fruits`, `init_hp`, `max_steps`) for the **NEXT training stage**.
3. A new `wall_layout` (list of [x, y] coordinates) for the **NEXT training stage**. Design a layout that seems appropriate given the agent's performance (e.g., add obstacles if it succeeds too easily, simplify if it dies to walls often).

{tunable_ranges_str}

**Output Format:**
Provide ONLY a single JSON object containing three keys: `reward_config`, `next_stage_env_params` (for scalar values), and `next_stage_wall_layout` (list of `[x, y]` pairs).
Example:
```json
{{
  "reward_config": {{
    "food_reward": 1.5, "death_penalty": -1.2, "step_penalty": -0.02",
    "center_bonus": 0.1, "loop_penalty": -0.4", "wall_follow_penalty": -0.05,
    "exploration_bonus": 0.05
  }},
  "next_stage_env_params": {{
    "gs": 12, "num_fruits": 2, "init_hp": 80, "max_steps": 1200
  }},
  "next_stage_wall_layout": [
    [3, 3], [3, 4], [3, 5],
    [8, 3], [8, 4], [8, 5]
  ]
}}
```

**Guidelines:**
- Adjust rewards aggressively based on metrics.
- Keep suggested scalar env params within specified ranges.
- Generate a `wall_layout` appropriate for the suggested `gs`. Use between 0 and {MAX_WALLS_SUGGESTION} wall coordinates. Ensure coordinates are valid `[x, y]` pairs within the suggested grid size (`gs`).
- Ensure the wall layout doesn't make the map impossible (e.g., completely blocking off areas).
- Provide ONLY the single JSON output. Do NOT include explanations.

**Provide the JSON output here:**

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
    # Ensure the prompt is correctly placed within the API structure
    data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7}} # Slightly higher temp might help creative map gen

    parsed_response = None # Store the final parsed JSON (reward + env params)
    try:
        print(f"[Callback] Sending request to Gemini API: {LLM_API_URL}")
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=90) # Increase timeout slightly

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
            # Add check for finishReason
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            if finish_reason not in ['STOP', 'MAX_TOKENS']:
                 print(f"[Callback] Warning: Candidate finish reason: {finish_reason}")
                 if 'safetyRatings' in candidate:
                      print(f"Safety Ratings: {candidate['safetyRatings']}")

            if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                llm_response_text = candidate['content']['parts'][0]['text']
                print(f"[Callback] LLM Extracted Response Text:\n{llm_response_text}\n")

                # --- More robust JSON parsing ---
                json_match = llm_response_text.strip()
                # Remove markdown fences first
                if json_match.startswith("```json"):
                    json_match = json_match[len("```json"):].strip()
                elif json_match.startswith("```"):
                     json_match = json_match[len("```"):].strip()
                if json_match.endswith("```"):
                    json_match = json_match[:-len("```")].strip()

                # Attempt to parse the cleaned string
                try:
                    parsed_response = json.loads(json_match)
                    # Basic validation of structure
                    if not isinstance(parsed_response, dict) or \
                       "reward_config" not in parsed_response or \
                       "next_stage_env_params" not in parsed_response or \
                       "next_stage_wall_layout" not in parsed_response or \
                       not isinstance(parsed_response["reward_config"], dict) or \
                       not isinstance(parsed_response["next_stage_env_params"], dict) or \
                       not isinstance(parsed_response["next_stage_wall_layout"], list):
                        print("[Callback] Error: Parsed JSON does not match expected structure (reward_config, next_stage_env_params, next_stage_wall_layout).")
                        parsed_response = None # Invalidate if structure is wrong
                except json.JSONDecodeError as json_e:
                    print(f"[Callback] Failed to parse JSON directly: {json_e}")
                    # Optional: Fallback if needed, but strict format was requested
                    # json_start = llm_response_text.find('{')
                    # json_end = llm_response_text.rfind('}') + 1
                    # if json_start != -1 and json_end != 0:
                    #      try: parsed_response = json.loads(llm_response_text[json_start:json_end]) ...
                    parsed_response = None # Strict format failed

            else:
                 print("[Callback] Warning: No 'content' or 'parts' in candidate.")
        else:
             print("[Callback] Warning: No candidates found in Gemini response.")
             if 'promptFeedback' in response_json and 'blockReason' in response_json['promptFeedback']:
                   print(f"Prompt blocked. Reason: {response_json['promptFeedback']['blockReason']}")

    except requests.exceptions.Timeout:
         print("[Callback] Error: Request to Gemini API timed out.")
    except requests.exceptions.RequestException as e:
        print(f"[Callback] Error calling Gemini API: {e}")
        if e.response is not None: pass
    except json.JSONDecodeError as e: # Should be caught by inner try-except now
         print(f"[Callback] Failed to parse JSON from LLM response: {e}")
         # print(f"LLM Raw Response was: {llm_response_text if 'llm_response_text' in locals() else 'N/A'}")
    except Exception as e:
        print(f"[Callback] An unexpected error occurred during LLM call: {e}")

    return parsed_response # Return the full parsed dict (or None)


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
    def __init__(self, check_freq_episodes: int, initial_env_params: dict, verbose=1):
        super().__init__(verbose)
        self.check_freq_episodes = check_freq_episodes
        self.total_episodes_completed = 0
        self.collected_episode_stats = []
        self.reward_config_history = []
        self.llm_call_count = 0
        # Store the full initial params, including potential wall_layout
        self.current_env_params = initial_env_params.copy()
        print(f"[Callback] Initialized. LLM Freq: {check_freq_episodes} eps.")
        print(f"[Callback] Initial Env Params: {self.current_env_params}")
        print(f"[Callback] Initial Reward Cfg: {metrics_collector.current_reward_config}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For Higgins, the environment metadata is stored in the local variables.
        """
        # Check for finished episodes in parallel environments
        newly_finished_episodes = 0
        infos_to_process = []
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                if "episode_stats" in info:
                    self.total_episodes_completed += 1
                    newly_finished_episodes += 1
                    infos_to_process.append(info["episode_stats"]) # Store stats
                    if self.verbose > 1:
                         print(f"[Callback] Env {i} finished episode {self.total_episodes_completed}.") # Less verbose log

        # Process collected stats if any episodes finished
        if newly_finished_episodes > 0:
             self.collected_episode_stats.extend(infos_to_process)

             # Check if enough total episodes have passed to trigger LLM
             # Use >= to ensure it triggers even if multiple finish at once crossing the threshold
             if self.total_episodes_completed >= self.check_freq_episodes * (self.llm_call_count + 1):
                  if self.verbose > 0:
                       print(f"\n--- [Callback] Total Episodes {self.total_episodes_completed} >= {self.check_freq_episodes * (self.llm_call_count + 1)}: Triggering LLM Reward Update ---")
                  # Check if we have collected enough stats for this window
                  if len(self.collected_episode_stats) >= self.check_freq_episodes // 2: # Heuristic: require at least half the target eps collected
                       self._trigger_llm_update()
                       self.collected_episode_stats = [] # Clear stats after successful trigger
                  else:
                       if self.verbose > 0:
                            print(f"[Callback] Threshold reached, but only {len(self.collected_episode_stats)}/{self.check_freq_episodes} eps collected. Waiting for more data.")

        # Log current reward config periodically
        log_freq_steps = 1000
        if self.num_timesteps > 0 and self.num_timesteps % log_freq_steps == 0:
             current_config = metrics_collector.current_reward_config
             log_dict = {f"reward_config/{k}": v for k, v in current_config.items()}
             log_dict.update({f"env_params/{k}": v for k, v in self.current_env_params.items() if k in TUNABLE_ENV_PARAMS})
             # Log current wall count
             log_dict["env_params/num_walls"] = len(self.current_env_params.get("wall_layout", []))
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
        summary["avg_episode_length"] = np.mean(agg["length"]) if agg["length"] else 0
        summary["avg_food_per_episode"] = np.mean(agg["food_eaten"]) if agg["food_eaten"] else 0
        valid_spf = [s for s in agg["steps_per_food"] if np.isfinite(s)]
        summary["avg_steps_per_food"] = np.mean(valid_spf) if valid_spf else 0
        summary["avg_max_snake_length"] = np.mean(agg["max_length"]) if agg["max_length"] else 0
        summary["avg_map_coverage_pct"] = np.mean(agg["map_coverage"]) * 100 if agg["map_coverage"] else 0
        summary["avg_center_visits"] = np.mean(agg["center_visits"]) if agg["center_visits"] else 0
        summary["looping_rate_pct"] = (looping_count / num_episodes) * 100 if num_episodes > 0 else 0
        summary["avg_action_entropy"] = np.mean(agg["action_entropy"]) if agg["action_entropy"] else 0
        summary["avg_turns_per_episode"] = np.mean(agg["turns"]) if agg["turns"] else 0


        total_deaths = sum(death_causes.values())
        summary["death_wall_pct"] = death_causes.get("wall", 0) / total_deaths * 100 if total_deaths > 0 else 0
        summary["death_self_pct"] = death_causes.get("self", 0) / total_deaths * 100 if total_deaths > 0 else 0
        # Group timeout/won/other
        other_deaths = total_deaths - death_causes.get("wall", 0) - death_causes.get("self", 0)
        summary["death_other_pct"] = other_deaths / total_deaths * 100 if total_deaths > 0 else 0

        return summary


    def _trigger_llm_update(self):
        """Aggregates metrics, calls LLM, updates global config, logs suggestions."""
        self.llm_call_count += 1
        aggregated_metrics = self._aggregate_metrics()

        if not aggregated_metrics:
            if self.verbose > 0: print("[Callback] No episode stats collected. Skipping LLM.")
            return

        current_reward_config = metrics_collector.current_reward_config.copy()
        current_wall_layout = self.current_env_params.get("wall_layout") # Get current layout

        # Log aggregated metrics to WandB
        wandb.log({f"llm_metrics/{k}": v for k, v in aggregated_metrics.items()}, step=self.num_timesteps)
        wandb.log({"llm_call_iteration": self.llm_call_count}, step=self.num_timesteps)

        # Record history entry (including current env params)
        history_entry = {
            "config_before": current_reward_config,
            "env_params_before": self.current_env_params.copy(),
            "metrics_used": aggregated_metrics.copy(),
            "llm_iteration": self.llm_call_count,
            "global_step": self.num_timesteps,
            "total_episodes": self.total_episodes_completed,
            "suggestion_details": {}
        }

        # Format prompt (passing current env params and history)
        prompt = format_llm_prompt(
            aggregated_metrics,
            current_reward_config,
            self.current_env_params,
            current_wall_layout, # Pass current wall layout
            self.reward_config_history
        )
        if not prompt:
            print("[Callback] Could not format LLM prompt.")
            self.reward_config_history.append(history_entry)
            return

        llm_suggestion = call_gemini_api(prompt)
        suggested_env_params = None
        suggested_wall_layout = None

        if llm_suggestion: # Check if suggestion is not None
            new_reward_config = llm_suggestion.get("reward_config")
            suggested_env_params = llm_suggestion.get("next_stage_env_params")
            suggested_wall_layout = llm_suggestion.get("next_stage_wall_layout")

            # 1. Update reward config immediately
            if new_reward_config and isinstance(new_reward_config, dict):
                metrics_collector.update_config(new_reward_config)
                history_entry["config_after"] = metrics_collector.current_reward_config.copy()
                # Log the *new* config values immediately
                log_dict_after = {f"reward_config/{k}": v for k, v in metrics_collector.current_reward_config.items()}
                wandb.log(log_dict_after, step=self.num_timesteps)
            else:
                history_entry["config_after"] = current_reward_config # Config didn't change
                print("[Callback] LLM response missing or invalid 'reward_config'.")

            # 2. Validate and Log suggested env params (DO NOT apply them now)
            validated_env_suggestions = {}
            if suggested_env_params and isinstance(suggested_env_params, dict):
                print(f"[Callback] LLM Suggested Env Params for NEXT stage: {suggested_env_params}")
                for key, value in suggested_env_params.items():
                     if key in TUNABLE_ENV_PARAMS:
                          try:
                               param_type = TUNABLE_ENV_PARAMS[key]['type']
                               param_range = TUNABLE_ENV_PARAMS[key]['range']
                               if param_type == 'int':
                                    val = int(value)
                                    val = max(param_range[0], min(val, param_range[1])) # Clamp
                               elif param_type == 'float': # Add if needed
                                    val = float(value)
                                    val = max(param_range[0], min(val, param_range[1]))
                               else: val = value # Keep as is if type unknown
                               validated_env_suggestions[key] = val
                          except (ValueError, TypeError):
                               print(f"Warning: Invalid type/value for suggested env param '{key}': {value}")
                               validated_env_suggestions[key] = "INVALID" # Mark as invalid
                     else:
                          validated_env_suggestions[key] = value # Keep untracked suggestions
                wandb.log({f"llm_suggestions/env_{k}": v for k, v in validated_env_suggestions.items() if v != "INVALID"}, step=self.num_timesteps)
                history_entry["suggestion_details"]["next_stage_env_params"] = validated_env_suggestions
            else:
                print("[Callback] LLM response missing or invalid 'next_stage_env_params'.")
                history_entry["suggestion_details"]["next_stage_env_params"] = None

            # 3. Validate and log suggested wall layout
            validated_wall_layout = None
            if suggested_wall_layout is not None and isinstance(suggested_wall_layout, list):
                 print(f"[Callback] LLM Suggested Wall Layout: {suggested_wall_layout}")
                 # Basic validation: ensure list of pairs, clamp coords to suggested gs (or current if none suggested)
                 suggested_gs = validated_env_suggestions.get("gs", self.current_env_params.get("gs", 10))
                 valid_walls = []
                 for item in suggested_wall_layout[:MAX_WALLS_SUGGESTION]: # Limit number of walls
                      if isinstance(item, list) and len(item) == 2:
                           try:
                                x, y = int(item[0]), int(item[1])
                                # Clamp coordinates to be within the suggested grid
                                x = max(0, min(x, suggested_gs - 1))
                                y = max(0, min(y, suggested_gs - 1))
                                valid_walls.append([x, y])
                           except (ValueError, TypeError): pass # Ignore invalid pairs
                 validated_wall_layout = valid_walls
                 wandb.log({"llm_suggestions/env_num_walls": len(validated_wall_layout)}, step=self.num_timesteps)
                 # Note: Logging the full layout list might be too much for wandb metrics, store in history
            else:
                 print("[Callback] LLM missing/invalid 'next_stage_wall_layout'.")
            history_entry["suggestion_details"]["next_stage_wall_layout"] = validated_wall_layout

        else:
             history_entry["config_after"] = current_reward_config # Config didn't change
             history_entry["suggestion_details"]["next_stage_env_params"] = None
             history_entry["suggestion_details"]["next_stage_wall_layout"] = None
             print("[Callback] LLM did not return a valid overall JSON suggestion.")

        self.reward_config_history.append(history_entry)

        # Save history periodically
        save_freq_llm_calls = 5
        if self.llm_call_count % save_freq_llm_calls == 0:
             self._save_history()

    def _save_history(self, final=False):
         """Saves the reward config history to JSON."""
         suffix = "final" if final else f"iter_{self.llm_call_count}"
         filename_local = f"reward_evolution_{suffix}.json"
         wandb_dir = wandb.run.dir if wandb.run else "."
         try:
              # Ensure save directory exists
              save_dir = "." if not wandb.run else wandb_dir
              if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
              full_path_local = os.path.join(save_dir, filename_local)

              with open(full_path_local, 'w') as f:
                   # Use default=str for potential numpy types if any creep in
                   json.dump(self.reward_config_history, f, indent=2, default=str)
              if self.verbose > 0:
                  print(f"[Callback] Reward evolution data saved to {full_path_local}")
              if wandb.run: wandb.save(full_path_local, base_path=wandb_dir, policy="now")
         except Exception as e:
              print(f"[Callback] Error saving reward evolution data: {e}")

    def _on_training_end(self):
        """Called at the end of training."""
        if self.verbose > 0:
            print("[Callback] Training ended. Saving final reward evolution history.")
        self._save_history(final=True)


# --- Main Training Function ---
def train():
    # --- Load initial game parameters ---
    config_file_path = f"{config_dir}/eval.json" # Or allow specifying stage config
    print(f"Loading initial game parameters from: {config_file_path}")
    try:
        with open(config_file_path, "r") as f:
            game_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file_path}")
        print("Please create an eval.json file in param_configs/.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {config_file_path}")
        return

    if 'rewards' in game_params: del game_params['rewards']

    # --- WandB Setup ---
    run = wandb.init(
        project="snake-rl-llm-map-design-v1", # New project name
        config={
            # Training Hyperparameters
            "policy_type": "CnnPolicy",
            "total_timesteps": 5_000_000, # Adjust as needed for longer stages
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
            "n_envs": 32 if USE_SUBPROC_VEC_ENV else 1,
            "seed": 42,
            "features_dim": 256,
            # Game Parameters (Log initial values used for this run)
            **game_params,
            # Initial Reward Configuration
            **metrics_collector.current_reward_config,
            # LLM Shaping Settings
            "llm_call_frequency_episodes": LLM_CALL_FREQUENCY_EPISODES,
            "llm_model": LLM_MODEL,
            "vec_env_type": "SubprocVecEnv" if USE_SUBPROC_VEC_ENV else "DummyVecEnv",
            "tunable_env_params_info": TUNABLE_ENV_PARAMS
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    config = wandb.config

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.features_dim)
    )

    vec_env_cls = SubprocVecEnv if USE_SUBPROC_VEC_ENV else DummyVecEnv
    print(f"Using VecEnv type: {vec_env_cls.__name__}")

    # Filter game_params passed to env to only include valid ones
    valid_env_args = SnakeGameEnv.__init__.__code__.co_varnames
    current_game_params = {k: v for k, v in game_params.items() if k in valid_env_args}
    print(f"Initializing VecEnv with params: {current_game_params}")

    try:
        vec_env = make_vec_env(
            lambda: SnakeGameEnv(**current_game_params),
            n_envs=config.n_envs, seed=config.seed, vec_env_cls=vec_env_cls
        )
    except Exception as e:
        print(f"Error creating vectorized environment: {e}")
        if wandb.run: run.finish(exit_code=1) # Ensure wandb run finishes on error
        return # Exit if env creation fails

    # --- Model Setup ---
    # Check for checkpoint to load (implement if needed for multi-stage)
    # model_load_path = "path/to/previous_model.zip" # Example
    # if os.path.exists(model_load_path):
    #     print(f"Loading model from {model_load_path}")
    #     model = PPO.load(model_load_path, env=vec_env)
    #     # Optionally reset learning rate, buffer etc if needed
    #     # model.set_learning_rate(config.learning_rate)
    # else:
    #     print("No checkpoint found, creating new model.")
    model = PPO(
        config.policy_type, vec_env, policy_kwargs=policy_kwargs, verbose=1,
        device='cuda' if th.cuda.is_available() else 'cpu',
        tensorboard_log=log_dir, learning_rate=config.learning_rate, n_steps=config.n_steps,
        batch_size=config.batch_size, n_epochs=config.n_epochs, gamma=config.gamma,
        gae_lambda=config.gae_lambda, clip_range=config.clip_range, ent_coef=config.ent_coef,
        vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm, seed=config.seed,
    )

    # --- Callbacks ---
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=f"{model_dir}/{run.id}",
        model_save_freq=max(1, config.n_steps * config.n_envs * 10),
        log="all",
        verbose=2,
    )
    # Pass the *actual* params used to initialize envs
    llm_callback = LLMTriggerCallback(check_freq_episodes=config.llm_call_frequency_episodes,
                                      initial_env_params=current_game_params, verbose=1)
    callbacks = [wandb_callback, llm_callback]

    # --- Training ---
    print(f"Starting training stage for {config.total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=config.total_timesteps, progress_bar=True,
                     tb_log_name=f"PPO_LLM_MapDesign_{run.id}", reset_num_timesteps=True, callback=callbacks)
        model.save(f'{model_dir}/{run.id}/final_model')
        print("Training stage finished normally.")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(f'{model_dir}/{run.id}/interrupted_model')
    finally:
        print("Closing environment and finishing run...")
        if 'vec_env' in locals(): vec_env.close()
        if wandb.run: run.finish()
        print("Training script finished.")


if __name__ == "__main__":
    train()