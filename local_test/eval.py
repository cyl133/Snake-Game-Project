from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
from stable_baselines3 import PPO
import torch
import json
import time
import numpy as np # Import numpy for mean/std calculation later
from stable_baselines3.common.monitor import Monitor
from feature_extractor import CustomCNN

# action_map = {
#     0: 'stay',
#     1: 'left',
#     2: 'right'
# }

action_map = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }

# Load parameters
with open("param_configs/eval.json", "r") as f:
    game_params = json.load(f)
    # Convert old format to new format
    if 'rewards' in game_params:
        # Create reward_config from the old rewards format
        reward_config = {
            "food_reward": game_params['rewards'].get('SnakeState.ATE', 21.0),
            "death_penalty": game_params['rewards'].get('SnakeState.DED', -10.0),
            "step_penalty": game_params['rewards'].get('SnakeState.OK', -0.4),
            "center_bonus": 0.0,
            "loop_penalty": 0.0,
            "wall_follow_penalty": 0.0,
            "exploration_bonus": 0.0
        }
        # Remove the old rewards field
        del game_params['rewards']
        # Add new reward_config
        game_params['reward_config'] = reward_config

# Load the trained model
# Make sure this path points to the correct model you want to evaluate
model_path = "/Users/chengyueli/Snake-Game-Project-1/local_test/models/model.zip"
print(f"Loading model from: {model_path}")
model = PPO.load(model_path)

# --- IMPORTANT: Create env with render_mode='human' ---
# Remove reward_config from game_params if it exists to avoid TypeError
if "reward_config" in game_params:
    # We pass reward_config explicitly, so remove it from game_params to avoid duplication
    extracted_reward_config = game_params.pop("reward_config")
else:
    # Handle case where reward_config might not be in game_params (e.g., older format)
    # This assumes reward_config was created above if 'rewards' key existed.
    if 'reward_config' not in locals():
         raise ValueError("Could not find reward configuration in game_params")
    extracted_reward_config = reward_config # Use the one created from 'rewards'

# Also remove 'rewards' key if it exists
if "rewards" in game_params:
    del game_params["rewards"]

# Create a new environment instance for evaluation with the reward_config and render_mode
print("Creating environment with render_mode='human'")
env = Monitor(SnakeGameEnv(**game_params, reward_config=extracted_reward_config, render_mode='human'))

# --- Comment out evaluate_policy as we use the manual loop ---
# print("Skipping evaluate_policy, using manual loop for rendering.")
# rew, std = evaluate_policy(model, env, n_eval_episodes=10, render=False, return_episode_rewards=False, warn=True, deterministic=False)


# --- Manual Evaluation Loop with Rendering ---
num_episodes = 10 # Adjust as needed
all_action_probs = []
episode_rewards = []
episode_lengths = []

print(f"\n--- Starting Manual Evaluation Loop ({num_episodes} episodes) ---")

for episode in range(num_episodes):
    obs, info = env.reset() # Get initial observation and info
    done = False
    terminated = False
    truncated = False
    current_reward = 0
    current_length = 0
    print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

    while not done:
        # Render the current state *before* taking the action
        env.render()

        with torch.no_grad():
            tensor_obs, _ = model.policy.obs_to_tensor(obs)
            action_dist = model.policy.get_distribution(tensor_obs)
            # Use .probs for distribution probabilities
            probs = action_dist.distribution.probs.cpu().numpy()[0]
            all_action_probs.append(probs.tolist())

        paired_probs = [(action_map[i], round(prob, ndigits=3)) for i, prob in enumerate(probs)]
        print(f"Step: {current_length}, Action Dist: {paired_probs}")

        action, _ = model.predict(obs, deterministic=False) # Use deterministic=False for eval
        print(f"Action Selected: {action_map[int(action)]}")

        obs, reward, terminated, truncated, info = env.step(int(action)) # Get Gymnasium outputs
        done = terminated or truncated # Check termination conditions

        current_reward += reward
        current_length += 1

    # Render the final state after loop ends
    print(f"Episode {episode + 1} finished. Final State:")
    env.render()
    # Retrieve final stats from Monitor wrapper's info dict
    ep_info = info.get("episode")
    if ep_info:
        print(f"  Monitor Reward: {ep_info['r']:.2f}, Monitor Length: {ep_info['l']}")
        episode_rewards.append(ep_info['r'])
        episode_lengths.append(ep_info['l'])
    else:
        # Fallback if Monitor info isn't available for some reason
        print(f"  Manual Reward: {current_reward:.2f}, Manual Length: {current_length}")
        episode_rewards.append(current_reward)
        episode_lengths.append(current_length)

    time.sleep(0.5)  # Pause a bit longer at the end of an episode

env.close() # Close the environment window

# Calculate and print mean/std after the loop
if episode_rewards:
    mean_reward_manual = np.mean(episode_rewards)
    std_reward_manual = np.std(episode_rewards)
    mean_length_manual = np.mean(episode_lengths)
    std_length_manual = np.std(episode_lengths)
    print("\n--- Evaluation Summary (Manual Loop) ---")
    print(f"Ran {len(episode_rewards)} episodes.")
    print(f"Mean Reward: {mean_reward_manual:.2f} +/- {std_reward_manual:.2f}")
    print(f"Mean Length: {mean_length_manual:.2f} +/- {std_length_manual:.2f}")
else:
    print("\n--- No episodes completed for summary ---")

# print(f"Original evaluate_policy: Mean Reward: {rew:.2f}, Std Reward: {std:.2f}") # Print if you uncomment evaluate_policy