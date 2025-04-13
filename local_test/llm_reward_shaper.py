import json
import numpy as np
import requests
import time
import os
from snake_game import SnakeState
from typing import Dict

# Default reward configuration - to be passed to environments directly
DEFAULT_REWARD_CONFIG = {
    "food_reward": 20.0,
    "death_penalty": -10.0,
    "step_penalty": -0.4,
    "center_bonus": 0.0,
    "loop_penalty": 0.0,
    "wall_follow_penalty": 0.0,
    "exploration_bonus": 0.0,
    "consecutive_food_bonus": 0.0,
    "distance_reduction_reward": 0.0,
    "wall_avoidance_bonus": 0.0
}

def get_reward_for_step(config: Dict, snake_condition: SnakeState, is_looping: bool = False,
                       in_center: bool = False, near_wall: bool = False,
                       unique_cell: bool = False) -> float:
    """Calculate reward for a step based on provided config."""
    reward = 0.0
    if snake_condition == SnakeState.ATE:
        reward += config.get("food_reward", 0.0)
    elif snake_condition == SnakeState.DED:
        reward += config.get("death_penalty", 0.0)
    else:  # SnakeState.OK
        reward += config.get("step_penalty", 0.0)

    if is_looping:
        reward += config.get("loop_penalty", 0.0)
    if in_center:
        reward += config.get("center_bonus", 0.0)
    if near_wall:
        reward += config.get("wall_follow_penalty", 0.0)
    if unique_cell:
        reward += config.get("exploration_bonus", 0.0)
    
    # Note: Other components like consecutive_food_bonus, distance_reduction_reward,
    # and wall_avoidance_bonus are handled directly in the gym_env.py step method
    return reward

# API configuration
GOOGLE_API_KEY = "AIzaSyANDTYyGq3EgFwctwRjlddZvqQhIfEnGH0"
LLM_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
LLM_MODEL = "gemini-2.0-flash"
LLM_API_URL = f"{LLM_API_URL_BASE}{LLM_MODEL}:generateContent?key={GOOGLE_API_KEY}"

class GlobalRewardConfig:
    """
    Holds the current dynamic reward configuration, updateable by the LLM callback.
    Calculates reward for a step based on the current config.
    """
    def __init__(self):
        # Default reward configuration
        self.current_reward_config = DEFAULT_REWARD_CONFIG.copy()
        # self.reward_config_history = []
        print(f"Initialized GlobalRewardConfig with: {self.current_reward_config}")

    def update_config(self, new_config: Dict):
        """Updates the reward config, called by the LLM trigger callback."""
        updated_keys = 0
        for key, value in new_config.items():
            if key in self.current_reward_config:
                try:
                    value = float(value)
                    if self.current_reward_config[key] != value:
                        self.current_reward_config[key] = value
                        updated_keys += 1
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value type for key '{key}': {value}. Keeping previous value.")
        if updated_keys > 0:
             print(f"[GlobalRewardConfig] Updated reward config: {self.current_reward_config}")
        else:
             print("[GlobalRewardConfig] Update called, but no valid changes detected.")

    def get_reward_for_step(self, snake_condition: SnakeState, is_looping: bool = False,
                           in_center: bool = False, near_wall: bool = False,
                           unique_cell: bool = False) -> float:
        """Calculate reward for a step based on current config."""
        # --- Reward calculation logic remains the same ---
        reward = 0.0
        if snake_condition == SnakeState.ATE:
            reward += self.current_reward_config["food_reward"]
        elif snake_condition == SnakeState.DED:
            reward += self.current_reward_config["death_penalty"]
        else:  # SnakeState.OK
            reward += self.current_reward_config["step_penalty"]

        if is_looping:
            reward += self.current_reward_config["loop_penalty"]
        if in_center:
            reward += self.current_reward_config["center_bonus"]
        if near_wall:
            reward += self.current_reward_config["wall_follow_penalty"]
        if unique_cell:
            reward += self.current_reward_config["exploration_bonus"]
        return reward

# Global instance for easy access by Env and Callback
# Renamed class for clarity
metrics_collector = GlobalRewardConfig()

# --- Removed all previous MetricsCollector methods for counting, aggregation, LLM calls ---
# --- They will now live in the Training Callback --- 