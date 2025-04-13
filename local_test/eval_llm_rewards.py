import json
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv


def evaluate_and_visualize(model_path, params_path, num_episodes=10, render=True):
    """
    Evaluate a trained model and visualize its performance.
    
    Args:
        model_path: Path to the trained model file
        params_path: Path to the parameters JSON file
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation
    """
    # Load game parameters
    with open(params_path, "r") as f:
        game_params = json.load(f)
    
    # Keep rewards from eval.json
    reward_config = game_params.get("rewards")
    if not reward_config:
        raise ValueError("No reward configuration found in eval.json")
    
    # Remove the 'rewards' key as it's no longer used by the Env's __init__
    if 'rewards' in game_params:
        del game_params['rewards']
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create a test environment
    env = SnakeGameEnv(**game_params, reward_config=reward_config, render_mode="human" if render else None)
    
    # Storage for metrics
    episode_rewards = []
    episode_lengths = []
    foods_eaten = []
    max_snake_lengths = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                
        # Extract final stats from the nested dict if available
        final_stats = info.get("episode_stats", {})
        episode_rewards.append(total_reward)
        episode_lengths.append(final_stats.get("length", steps)) # Use final length if available
        foods_eaten.append(final_stats.get("food_eaten", 0))
        max_snake_lengths.append(final_stats.get("max_length", 0))
        
        print(f"  Steps: {final_stats.get('length', steps)}")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Food eaten: {final_stats.get('food_eaten', 0)}")
        print(f"  Max snake length: {final_stats.get('max_length', 0)}")
    
    # Close the environment
    env.close()
    
    # Print summary statistics
    print("\n--- Evaluation Summary ---")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average food eaten: {np.mean(foods_eaten):.2f} ± {np.std(foods_eaten):.2f}")
    print(f"Average max snake length: {np.mean(max_snake_lengths):.2f} ± {np.std(max_snake_lengths):.2f}")
    
    # Create visualizations if we have enough data
    if len(episode_rewards) > 1:
        plt.figure(figsize=(12, 10))
        
        # Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        
        # Episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths')
        
        # Food eaten
        plt.subplot(2, 2, 3)
        plt.plot(foods_eaten)
        plt.xlabel('Episode')
        plt.ylabel('Food Eaten')
        plt.title('Food per Episode')
        
        # Max snake length
        plt.subplot(2, 2, 4)
        plt.plot(max_snake_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Max Length')
        plt.title('Max Snake Length')
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.show()


def plot_reward_evolution(evolution_file="reward_evolution_final.json"):
    """
    Plot how the reward function evolved over time based on LLM suggestions.
    Uses the new history structure saved by the callback.
    """
    if not os.path.exists(evolution_file):
         # Try finding the latest iteration file if final doesn't exist
         iter_files = sorted([f for f in os.listdir('.') if f.startswith('reward_evolution_iter_') and f.endswith('.json')],
                             key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
         if iter_files:
              evolution_file = iter_files[0]
              print(f"Final evolution file not found, using latest iteration: {evolution_file}")
         else:
              print(f"Error: Reward evolution file not found: {evolution_file}")
              return

    try:
        with open(evolution_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading reward evolution data from {evolution_file}: {e}")
        return

    if not data:
        print("No reward evolution data found")
        return

    # Extract iterations and reward components from the new structure
    iterations = [entry["llm_iteration"] for entry in data]
    global_steps = [entry["global_step"] for entry in data] # Use global step for x-axis

    # Get all unique reward component keys (check both before/after)
    all_keys = set()
    for entry in data:
        all_keys.update(entry["config_before"].keys())
        if "config_after" in entry:
             all_keys.update(entry["config_after"].keys())

    plt.figure(figsize=(14, 8))
    for key in sorted(all_keys):
        # Plot value *after* the LLM call for this iteration
        values_after = [entry.get("config_after", entry["config_before"]).get(key, 0) for entry in data]
        plt.plot(global_steps, values_after, marker='o', linestyle='-', label=key)

    plt.xlabel('Global Timestep')
    plt.ylabel('Reward Value')
    plt.title('Evolution of Reward Components (Value After LLM Call)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig('reward_evolution.png')
    plt.show()

    # Plot performance metrics over time
    plt.figure(figsize=(15, 10))
    metric_keys = set()
    for entry in data:
        if "metrics_used" in entry:
            metric_keys.update(entry["metrics_used"].keys())

    plot_metrics = [
        "avg_episode_length", "avg_food_per_episode", "avg_max_snake_length",
        "avg_map_coverage_pct", "looping_rate_pct", "avg_action_entropy"
    ]
    plot_metrics = [m for m in plot_metrics if m in metric_keys]

    num_plots = len(plot_metrics)
    for i, key in enumerate(plot_metrics):
        plt.subplot(num_plots, 1, i+1)
        # Get metrics that *led* to this LLM call iteration
        values = [entry["metrics_used"].get(key, 0) if "metrics_used" in entry else 0 for entry in data]
        plt.plot(global_steps, values, marker='o', linestyle='-')
        plt.ylabel(key.replace("_", " ").title())
        if i == num_plots - 1:
            plt.xlabel('Global Timestep (Metrics Leading to LLM Call)')
        else:
             plt.xticks([]) # Hide x-axis labels for upper plots
        plt.grid(True)
        plt.title(f"Evolution of {key.replace('_', ' ').title()}")

    plt.suptitle("Evolution of Aggregated Metrics Leading to LLM Calls", y=1.02)
    plt.tight_layout()
    plt.savefig('metrics_evolution.png')
    plt.show()


if __name__ == "__main__":
    # Default paths - update these as needed
    # Find the latest model automatically
    models_dir = "models_wandb"
    latest_model_path = None
    if os.path.exists(models_dir):
        all_runs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if all_runs:
             latest_run = max(all_runs, key=os.path.getmtime)
             model_files = [os.path.join(latest_run, f) for f in os.listdir(latest_run) if f.endswith(".zip")]
             if model_files:
                  latest_model_path = max(model_files, key=os.path.getmtime)
                  print(f"Found latest model: {latest_model_path}")

    if latest_model_path is None:
         print("Error: No model found in models_wandb directory.")
         exit()

    MODEL_PATH = latest_model_path
    PARAMS_PATH = "param_configs/eval.json"

    # First plot the reward and metrics evolution
    plot_reward_evolution() # Looks for reward_evolution_final.json or latest iter

    # Then evaluate the trained model
    evaluate_and_visualize(MODEL_PATH, PARAMS_PATH, num_episodes=10, render=False) # Render = False for quick eval 