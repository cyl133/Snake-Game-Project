import json
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
# Assume TUNABLE_ENV_PARAMS might be useful here, or define relevant keys directly
PLOT_ENV_PARAMS = ["gs", "num_fruits", "init_hp", "max_steps"] # Define which env params to plot


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
    env = SnakeGameEnv(**game_params, render_mode="human" if render else None)
    
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
    """Plots reward evolution, metrics, and suggested env params."""
    if not os.path.exists(evolution_file):
         iter_files = sorted([f for f in os.listdir('.') if f.startswith('reward_evolution_iter_') and f.endswith('.json')],
                             key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
         if iter_files: evolution_file = iter_files[0]; print(f"Using latest: {evolution_file}")
         else: print(f"Error: File not found: {evolution_file}"); return

    try:
        with open(evolution_file, 'r') as f: data = json.load(f)
    except Exception as e: print(f"Error loading data from {evolution_file}: {e}"); return
    if not data: print("No reward evolution data found"); return

    iterations = [entry["llm_iteration"] for entry in data]
    global_steps = [entry["global_step"] for entry in data]

    # --- Plot Reward Evolution ---
    reward_keys = set()
    for entry in data:
        reward_keys.update(entry["config_before"].keys())
        if "config_after" in entry: reward_keys.update(entry["config_after"].keys())

    plt.figure(figsize=(14, 8))
    for key in sorted(reward_keys):
        values_after = [entry.get("config_after", entry["config_before"]).get(key, 0) for entry in data]
        plt.plot(global_steps, values_after, marker='o', linestyle='-', label=key)
    plt.xlabel('Global Timestep'); plt.ylabel('Reward Value')
    plt.title('Evolution of Reward Components (Value After LLM Call)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig('reward_evolution.png'); plt.show()

    # --- Plot Metrics Evolution ---
    metric_keys = set()
    for entry in data:
        if "metrics_used" in entry: metric_keys.update(entry["metrics_used"].keys())
    plot_metrics = ["avg_episode_length", "avg_food_per_episode", "avg_max_snake_length",
                    "avg_map_coverage_pct", "looping_rate_pct", "avg_action_entropy"]
    plot_metrics = [m for m in plot_metrics if m in metric_keys]
    num_plots_metrics = len(plot_metrics)

    if num_plots_metrics > 0:
         plt.figure(figsize=(15, 3 * num_plots_metrics)) # Adjust height
         for i, key in enumerate(plot_metrics):
              plt.subplot(num_plots_metrics, 1, i+1)
              values = [entry["metrics_used"].get(key, 0) if "metrics_used" in entry else 0 for entry in data]
              plt.plot(global_steps, values, marker='.', linestyle='-') # Use dots for metrics
              plt.ylabel(key.replace("_", " ").title())
              if i == num_plots_metrics - 1: plt.xlabel('Global Timestep (Metrics Leading to LLM Call)')
              else: plt.xticks([])
              plt.grid(True); plt.title(f"Metric: {key.replace('_', ' ').title()}")
         plt.suptitle("Evolution of Aggregated Metrics Leading to LLM Calls", y=1.01)
         plt.tight_layout(rect=[0, 0, 1, 1]); plt.savefig('metrics_evolution.png'); plt.show()
    else:
         print("No metrics found in history to plot.")


    # --- Plot Suggested Env Param Evolution ---
    env_param_keys = set()
    for entry in data:
         if "suggestion_details" in entry and entry["suggestion_details"].get("next_stage_env_params"):
              env_param_keys.update(entry["suggestion_details"]["next_stage_env_params"].keys())
    # Filter for keys we care about plotting
    plot_env_keys = [k for k in PLOT_ENV_PARAMS if k in env_param_keys]
    num_plots_env = len(plot_env_keys)

    if num_plots_env > 0:
         plt.figure(figsize=(15, 3 * num_plots_env)) # Adjust height
         for i, key in enumerate(plot_env_keys):
              plt.subplot(num_plots_env, 1, i+1)
              # Get suggested value for this key at each iteration
              values = []
              valid_steps = []
              for entry in data:
                   suggestion = entry.get("suggestion_details", {}).get("next_stage_env_params")
                   if suggestion and key in suggestion and suggestion[key] != "INVALID":
                        values.append(suggestion[key])
                        valid_steps.append(entry["global_step"])
              if values: # Only plot if there are valid suggestions
                   plt.plot(valid_steps, values, marker='x', linestyle='--', label=key) # Use crosses for suggestions
              plt.ylabel(f"Suggested {key}")
              if i == num_plots_env - 1: plt.xlabel('Global Timestep (LLM Suggestion Point)')
              else: plt.xticks([])
              plt.grid(True); plt.title(f"LLM Suggestion for Next Stage: {key}")
              plt.legend() # Show key name

         plt.suptitle("LLM Suggestions for Next Stage Environment Parameters", y=1.01)
         plt.tight_layout(rect=[0, 0, 1, 1]); plt.savefig('env_param_suggestions.png'); plt.show()
    else:
         print("No environment parameter suggestions found in history to plot.")


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

    # Plot evolution first
    plot_reward_evolution()

    # Then evaluate model (using params from PARAMS_PATH)
    evaluate_and_visualize(MODEL_PATH, PARAMS_PATH, num_episodes=10, render=False) 