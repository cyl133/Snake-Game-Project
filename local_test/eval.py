from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
from stable_baselines3 import PPO


# Load the trained model
model = PPO.load("models/ppo_snake2.0.zip")

# Create a new environment instance for evaluation
env = SnakeGameEnv(num_snakes=1, num_teams=1, render_mode='human')

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
