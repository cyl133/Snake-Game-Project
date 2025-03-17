from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
from stable_baselines3 import PPO


# Load the trained model
model = PPO.load("models/ppo_snake3.11.zip")

# Create a new environment instance for evaluation
env = SnakeGameEnv(num_snakes=1, num_teams=1, render_mode='human')

# Evaluate the model
rew, std = evaluate_policy(model, env, n_eval_episodes=50, render=True, return_episode_rewards=False, warn=False)

print(f"Mean Reward: {rew:.2f}, Std Reward: {std:.2f}")