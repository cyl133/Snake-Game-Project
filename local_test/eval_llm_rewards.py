import json
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
from llm_reward_shaper import metrics_collector


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
                
        # Collect episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        foods_eaten.append(info.get("food_eaten", 0))
        max_snake_lengths.append(info.get("max_snake_length", 0))
        
        print(f"  Steps: {steps}")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Food eaten: {info.get('food_eaten', 0)}")
        print(f"  Max snake length: {info.get('max_snake_length', 0)}")
    
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


def plot_reward_evolution(evolution_file="reward_evolution_latest.json"):
    """
    Plot how the reward function evolved over time based on LLM suggestions.
    
    Args:
        evolution_file: Path to the JSON file with reward evolution data
    """
    try:
        with open(evolution_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading reward evolution data: {e}")
        return
    
    if not data:
        print("No reward evolution data found")
        return
    
    # Extract iterations and reward components
    iterations = [entry["iteration"] for entry in data]
    
    # Get all unique reward component keys
    all_keys = set()
    for entry in data:
        all_keys.update(entry["config"].keys())
    
    # Plot each reward component over time
    plt.figure(figsize=(12, 8))
    for key in sorted(all_keys):
        values = [entry["config"].get(key, 0) for entry in data]
        plt.plot(iterations, values, marker='o', label=key)
    
    plt.xlabel('LLM Call Iteration')
    plt.ylabel('Reward Value')
    plt.title('Evolution of Reward Components')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_evolution.png')
    plt.show()
    
    # Plot performance metrics over time
    plt.figure(figsize=(15, 10))
    
    # Get all unique metrics keys
    metric_keys = set()
    for entry in data:
        if "metrics" in entry:
            metric_keys.update(entry["metrics"].keys())
    
    # Select relevant metrics to plot
    plot_metrics = [
        "avg_episode_length", 
        "avg_food_per_episode", 
        "max_snake_length", 
        "map_coverage_pct",
        "looping_rate_pct"
    ]
    
    plot_metrics = [m for m in plot_metrics if m in metric_keys]
    
    for i, key in enumerate(plot_metrics):
        plt.subplot(len(plot_metrics), 1, i+1)
        values = [entry["metrics"].get(key, 0) if "metrics" in entry else 0 for entry in data]
        plt.plot(iterations, values, marker='o')
        plt.ylabel(key)
        if i == len(plot_metrics) - 1:
            plt.xlabel('LLM Call Iteration')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics_evolution.png')
    plt.show()


if __name__ == "__main__":
    # Default paths - update these as needed
    MODEL_PATH = "models_wandb/latest/final_model.zip"  # Path to your trained model
    PARAMS_PATH = "param_configs/eval.json"
    
    # Check if model exists, if not search for most recent one
    if not os.path.exists(MODEL_PATH):
        # Look in models_wandb directory for the most recent model file
        models_dir = "models_wandb"
        if os.path.exists(models_dir):
            all_models = []
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".zip"):
                        all_models.append(os.path.join(root, file))
            
            if all_models:
                # Sort by modification time (most recent first)
                all_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                MODEL_PATH = all_models[0]
                print(f"Using most recent model: {MODEL_PATH}")
    
    # First plot the reward and metrics evolution
    plot_reward_evolution()
    
    # Then evaluate the trained model
    evaluate_and_visualize(MODEL_PATH, PARAMS_PATH, num_episodes=5, render=True) 