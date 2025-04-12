import json
import numpy as np
import requests
import time
import os
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple
from gymnasium.core import Env
from snake_game import SnakeState

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables for production.
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
GOOGLE_API_KEY = "AIzaSyANDTYyGq3EgFwctwRjlddZvqQhIfEnGH0" # Directly using the provided key (INSECURE)

# Use the Gemini Pro endpoint
LLM_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
LLM_MODEL = "gemini-pro" # Use Gemini Pro model
LLM_API_URL = f"{LLM_API_URL_BASE}{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"

METRICS_COLLECTION_FREQUENCY = 10  # Collect metrics every N episodes
LLM_CALL_FREQUENCY = 500  # Call LLM every M episodes
GRID_SIZE = 10  # Default grid size, should match your env

class MetricsCollector:
    """Collects detailed metrics about the Snake agent's performance."""
    
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.reset_metrics()
        
        # Default reward configuration
        self.current_reward_config = {
            "food_reward": 1.0,
            "death_penalty": -1.0,
            "step_penalty": -0.01,
            "center_bonus": 0.0,
            "loop_penalty": 0.0,
            "wall_follow_penalty": 0.0,
            "exploration_bonus": 0.0
        }
        
        # Episode counter
        self.episode_count = 0
        self.llm_call_count = 0
        
        # History of visited positions for detecting loops and coverage
        self.position_history = []
        self.visited_positions = set()
        
        # Track historical configs and their performance
        self.reward_config_history = []
        
    def reset_metrics(self):
        """Reset all metrics for a new collection window."""
        self.episode_lengths = []
        self.food_per_episode = []
        self.steps_per_food = []
        self.death_causes = {"wall": 0, "self": 0, "timeout": 0}
        self.max_snake_lengths = []
        self.center_region_visits = []
        self.actions_taken = []  # For action entropy
        self.turn_counts = []
        self.map_coverage = []
        self.looping_episodes = 0
        
        # Grid cell visit counts for heatmap
        self.visit_heatmap = np.zeros((self.grid_size, self.grid_size))
        
    def start_episode(self):
        """Call at the start of each episode."""
        self.position_history = []
        self.visited_positions = set()
        self.actions_this_episode = []
        self.turns_this_episode = 0
        self.center_visits_this_episode = 0
        
    def update_position(self, x: int, y: int, action: int):
        """Update with the snake's current position and action."""
        pos = (x, y)
        
        # Track position for coverage and loop detection
        self.position_history.append(pos)
        self.visited_positions.add(pos)
        
        # Update heatmap (Ensure indices are within bounds)
        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
             self.visit_heatmap[y, x] += 1
        
        # Track action for entropy calculation
        self.actions_this_episode.append(action)
        
        # Check if in center region (inner 1/3 of the grid)
        center_start = self.grid_size // 3
        center_end = self.grid_size - center_start
        if center_start <= x < center_end and center_start <= y < center_end:
            self.center_visits_this_episode += 1
            
        # Check for turns (changes in action)
        if len(self.actions_this_episode) >= 2:
            # Compare last two actions
            if self.actions_this_episode[-1] != self.actions_this_episode[-2]:
                 self.turns_this_episode += 1
                
    def detect_loops(self, window_size: int = 20, min_loop_length: int = 4) -> bool:
        """Check if the snake is looping in recent movements."""
        if len(self.position_history) < window_size + min_loop_length:
            return False
            
        # Check recent positions for repeated patterns
        recent_positions = self.position_history[-window_size:]
        for pattern_length in range(min_loop_length, window_size // 2 + 1):
            pattern = recent_positions[-pattern_length:]
            previous_segment = recent_positions[-(pattern_length*2):-pattern_length]
            if pattern == previous_segment:
                return True
                
        return False
        
    def end_episode(self, length: int, food_eaten: int, death_cause: str, max_length: int):
        """Call at the end of each episode with final results."""
        self.episode_count += 1
        
        # Record basic stats
        self.episode_lengths.append(length)
        self.food_per_episode.append(food_eaten)
        self.max_snake_lengths.append(max_length)
        
        # Calculate steps per food (avoid division by zero)
        steps_per_food = length / max(1, food_eaten)
        self.steps_per_food.append(steps_per_food)
        
        # Record death cause
        if death_cause in self.death_causes:
            self.death_causes[death_cause] += 1
            
        # Record map coverage
        coverage = len(self.visited_positions) / max(1, (self.grid_size * self.grid_size)) # Avoid division by zero
        self.map_coverage.append(coverage)
        
        # Record center visits
        self.center_region_visits.append(self.center_visits_this_episode)
        
        # Record turns
        self.turn_counts.append(self.turns_this_episode)
        
        # Record actions for entropy calculation
        self.actions_taken.extend(self.actions_this_episode)
        
        # Check if this was a looping episode
        if self.detect_loops():
            self.looping_episodes += 1
            
        # Check if it's time to call the LLM
        if self.episode_count > 0 and self.episode_count % LLM_CALL_FREQUENCY == 0:
             print(f"\n--- Episode {self.episode_count}: Triggering LLM Reward Update ---")
             self.call_llm_for_reward_update()
            
    def calculate_action_entropy(self) -> float:
        """Calculate the entropy of the action distribution."""
        if not self.actions_taken:
            return 0.0
            
        # Count occurrences of each action
        action_counts = defaultdict(int)
        for action in self.actions_taken:
            action_counts[action] += 1
            
        # Calculate probabilities
        total_actions = len(self.actions_taken)
        probs = [count / total_actions for count in action_counts.values()]
        
        # Calculate entropy: -sum(p * log2(p)) (using log base 2 for bits)
        entropy = -sum(p * np.log2(p) for p in probs if p > 0) # Add check p > 0
        return entropy
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a summary of collected metrics."""
        total_episodes = len(self.episode_lengths)
        if total_episodes == 0:  # No data yet
            return {}
            
        # Calculate summary statistics, handle potential division by zero
        avg_episode_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        avg_food_per_episode = np.mean(self.food_per_episode) if self.food_per_episode else 0
        # Filter out potential inf values if food_eaten was 0
        valid_steps_per_food = [s for s in self.steps_per_food if np.isfinite(s)]
        avg_steps_per_food = np.mean(valid_steps_per_food) if valid_steps_per_food else 0
        max_snake_length = max(self.max_snake_lengths) if self.max_snake_lengths else 0
        
        # Death cause distribution
        total_deaths = sum(self.death_causes.values())
        death_wall_pct = self.death_causes["wall"] / total_deaths * 100 if total_deaths > 0 else 0
        death_self_pct = self.death_causes["self"] / total_deaths * 100 if total_deaths > 0 else 0
        death_timeout_pct = self.death_causes["timeout"] / total_deaths * 100 if total_deaths > 0 else 0
        
        # Map coverage
        avg_map_coverage = np.mean(self.map_coverage) * 100 if self.map_coverage else 0
        
        # Center region visits
        avg_center_visits = np.mean(self.center_region_visits) if self.center_region_visits else 0
        
        # Looping behavior
        looping_rate = self.looping_episodes / total_episodes * 100 if total_episodes > 0 else 0
        
        # Action entropy
        action_entropy = self.calculate_action_entropy()
        
        # Turn frequency
        avg_turns = np.mean(self.turn_counts) if self.turn_counts else 0
        
        return {
            "avg_episode_length": round(avg_episode_length, 2),
            "avg_food_per_episode": round(avg_food_per_episode, 2),
            "avg_steps_per_food": round(avg_steps_per_food, 2),
            "max_snake_length": max_snake_length,
            "death_wall_pct": round(death_wall_pct, 2),
            "death_self_pct": round(death_self_pct, 2),
            "death_timeout_pct": round(death_timeout_pct, 2),
            "map_coverage_pct": round(avg_map_coverage, 2),
            "avg_center_visits": round(avg_center_visits, 2),
            "looping_rate_pct": round(looping_rate, 2),
            "action_entropy": round(action_entropy, 3),
            "avg_turns_per_episode": round(avg_turns, 2),
            "episodes_collected": total_episodes
        }
        
    def get_llm_prompt(self) -> str:
        """Generate a prompt for the LLM based on collected metrics."""
        metrics = self.get_metrics_summary()
        if not metrics:
            return ""
            
        # Format the prompt clearly for the LLM
        prompt = f"""
Analyze the behavior of a Snake RL agent based on these metrics from the last {metrics['episodes_collected']} episodes:

**Performance:**
- Average Episode Length: {metrics['avg_episode_length']} steps
- Average Food Eaten: {metrics['avg_food_per_episode']}
- Average Steps per Food: {metrics['avg_steps_per_food']}
- Maximum Snake Length Achieved: {metrics['max_snake_length']}

**Survival:**
- Death Causes: Wall ({metrics['death_wall_pct']}%), Self-Collision ({metrics['death_self_pct']}%), Timeout ({metrics['death_timeout_pct']}%)

**Exploration & Behavior:**
- Map Coverage: {metrics['map_coverage_pct']}%
- Average Center Visits per Episode: {metrics['avg_center_visits']}
- Looping Behavior Rate: {metrics['looping_rate_pct']}%
- Action Entropy: {metrics['action_entropy']} (Higher means more random actions)
- Average Turns per Episode: {metrics['avg_turns_per_episode']}

**Current Reward Function:**
```json
{json.dumps(self.current_reward_config, indent=2)}
```

**Task:**
Based ONLY on the metrics provided, suggest modifications to the reward function JSON below to encourage better performance (more food, longer survival) and exploration (higher coverage, less looping).

**Constraints:**
- Keep reward values between -5.0 and 5.0.
- Provide ONLY the updated JSON configuration. Do NOT include explanations or surrounding text.

**Provide the updated reward JSON here:**
```json
{{
  "food_reward": {self.current_reward_config.get('food_reward', 1.0)},
  "death_penalty": {self.current_reward_config.get('death_penalty', -1.0)},
  "step_penalty": {self.current_reward_config.get('step_penalty', -0.01)},
  "center_bonus": {self.current_reward_config.get('center_bonus', 0.0)},
  "loop_penalty": {self.current_reward_config.get('loop_penalty', 0.0)},
  "wall_follow_penalty": {self.current_reward_config.get('wall_follow_penalty', 0.0)},
  "exploration_bonus": {self.current_reward_config.get('exploration_bonus', 0.0)}
}}
```
"""
        return prompt.strip() # Remove leading/trailing whitespace
        
    def call_llm_for_reward_update(self):
        """Call the Gemini API to get reward function updates."""
        self.llm_call_count += 1
        
        prompt = self.get_llm_prompt()
        if not prompt:
            print("No metrics collected yet, skipping LLM call.")
            return
        
        # Record current performance before potential update
        performance_snapshot = {
            "config": self.current_reward_config.copy(),
            "metrics": self.get_metrics_summary(),
            "iteration": self.llm_call_count
        }
        self.reward_config_history.append(performance_snapshot)
        
        # Skip if API key is missing
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
             print("Google API key not set. Skipping LLM call.")
             # --- DEBUG PRINT: Show prompt even when skipping API call ---
             print("\n--- LLM PROMPT (SKIPPED) ---")
             print(prompt)
             print("---------------------------\n")
             # Reset metrics for the next cycle even if LLM is skipped
             self.reset_metrics()
             return
        
        # --- DEBUG PRINT: Show the prompt being sent ---
        print("\n--- LLM PROMPT ---")
        print(prompt)
        print("------------------\n")
        
        try:
            # Prepare Gemini API request data
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                # Optional: Add generation config (temperature, safety settings etc.)
                # "generationConfig": {
                #     "temperature": 0.7,
                #     "topK": 1,
                #     "topP": 1,
                #     "maxOutputTokens": 2048,
                # },
                # "safetySettings": [...] # Add safety settings if needed
            }
            
            print(f"Sending request to Gemini API: {LLM_API_URL}")
            response = requests.post(LLM_API_URL, headers=headers, json=data)

            # --- DEBUG PRINT: Show the raw response status and JSON ---
            print(f"\n--- LLM RAW RESPONSE (Status: {response.status_code}) ---")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2)) # Pretty print the JSON
            except json.JSONDecodeError:
                print("Could not decode JSON response:")
                print(response.text)
            print("------------------------------------\n")

            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Extract the response text from Gemini structure
            if 'candidates' in response_json and len(response_json['candidates']) > 0:
                 # Handle potential variations in response structure
                 candidate = response_json['candidates'][0]
                 if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                      llm_response_text = candidate['content']['parts'][0]['text']
                      # Current print statement for extracted text is already good
                      print(f"LLM Extracted Response Text:\n{llm_response_text}\n")
                      self.update_reward_config_from_llm(llm_response_text)
                 else:
                      print("Warning: Unexpected response structure from Gemini.")
                      # print(f"Full Response: {response_json}") # Already printed above

            else:
                 print("Warning: No candidates found in Gemini response.")
                 # print(f"Full Response: {response_json}") # Already printed above
                 # Handle cases where the response might be blocked due to safety settings
                 if 'promptFeedback' in response_json and 'blockReason' in response_json['promptFeedback']:
                       print(f"Prompt blocked. Reason: {response_json['promptFeedback']['blockReason']}")

        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            # Optionally, print response body if available
            if e.response is not None:
                 # Raw response already printed above in the success case debug block
                 pass
                 # print(f"Response status code: {e.response.status_code}")
                 # print(f"Response body: {e.response.text}")
        except Exception as e:
             print(f"An unexpected error occurred during LLM call: {e}")
        
        # Reset metrics for the next collection window regardless of LLM success/failure
        self.reset_metrics()
        print(f"Metrics reset for next {LLM_CALL_FREQUENCY} episodes.")
        
    def update_reward_config_from_llm(self, llm_response: str):
        """Parse the LLM response and update the reward configuration."""
        try:
            # Try to extract JSON strictly, assuming LLM follows instructions
            json_match = llm_response.strip()
            # Remove potential markdown code block fences
            if json_match.startswith("```json"):
                 json_match = json_match[len("```json"):].strip()
            if json_match.startswith("```"):
                 json_match = json_match[len("```"):].strip()
            if json_match.endswith("```"):
                 json_match = json_match[:-len("```")].strip()
            
            if not json_match.startswith("{") or not json_match.endswith("}"):
                 print("Warning: LLM response does not appear to be valid JSON object.")
                 # Fallback: try finding JSON boundaries if strict parsing fails
                 json_start = llm_response.find('{')
                 json_end = llm_response.rfind('}') + 1
                 if json_start != -1 and json_end != 0:
                      json_match = llm_response[json_start:json_end]
                 else:
                      print("Could not extract JSON from LLM response.")
                      return
            
            new_config = json.loads(json_match)
            
            updated_keys = 0
            # Validate and merge with existing config
            for key, value in new_config.items():
                if key in self.current_reward_config:
                    try:
                         # Ensure values are floats and within reasonable bounds
                         value = float(value)
                         # Using the -5.0 to 5.0 range mentioned in the prompt
                         value = max(min(value, 5.0), -5.0)
                         if self.current_reward_config[key] != value:
                              self.current_reward_config[key] = value
                              updated_keys += 1
                    except (ValueError, TypeError):
                         print(f"Warning: Invalid value type for key '{key}' from LLM: {value}. Keeping previous value.")
            
            if updated_keys > 0:
                print(f"Successfully updated reward config from LLM: {self.current_reward_config}")
            else:
                print("LLM response parsed, but no valid changes were made to the reward config.")
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM response: {e}")
            print(f"LLM Raw Response was: {llm_response}")
        except Exception as e:
            print(f"Error updating reward config: {e}")
        
    def get_reward_for_step(self, snake_condition: SnakeState, is_looping: bool = False, 
                           in_center: bool = False, near_wall: bool = False,
                           unique_cell: bool = False) -> float:
        """Calculate reward for a step based on current config."""
        reward = 0.0
        
        # Basic rewards based on snake condition
        if snake_condition == SnakeState.ATE:
            reward += self.current_reward_config["food_reward"]
        elif snake_condition == SnakeState.DED:
            reward += self.current_reward_config["death_penalty"]
        else:  # SnakeState.OK
            reward += self.current_reward_config["step_penalty"]
            
        # Additional shaping rewards
        if is_looping:
            reward += self.current_reward_config["loop_penalty"]
        if in_center:
            reward += self.current_reward_config["center_bonus"]
        if near_wall:
            reward += self.current_reward_config["wall_follow_penalty"]
        if unique_cell:
            reward += self.current_reward_config["exploration_bonus"]
            
        return reward
        
    def export_results(self, filename: str = "reward_evolution.json"):
        """Export the history of reward configs and their performance."""
        try:
             with open(filename, 'w') as f:
                  json.dump(self.reward_config_history, f, indent=2)
             print(f"Reward evolution data exported to {filename}")
        except Exception as e:
             print(f"Error exporting reward evolution data: {e}")


# Global instance for easy access
metrics_collector = MetricsCollector() 