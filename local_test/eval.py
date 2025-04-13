from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import SnakeGameEnv
from stable_baselines3 import PPO
import torch
import json
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
model = PPO.load("/Users/chengyueli/Snake-Game-Project-1/models/model.zip")

# Create a new environment instance for evaluation with the reward_config
env = Monitor(SnakeGameEnv(**game_params))

# Evaluate the model
rew, std = evaluate_policy(model, env, n_eval_episodes=10, render=True, return_episode_rewards=False, warn=True, deterministic=False)


# For getting the explicit actions probabilities, could be good for data and reporting
# num_episodes = 50
# all_action_probs = []

# for _ in range(num_episodes):
#     obs, _ = env.reset()
#     done = False

#     while not done:
#         with torch.no_grad():
#             tensor_obs, _ = model.policy.obs_to_tensor(obs)
#             action_dist = model.policy.get_distribution(tensor_obs)
#             action_probs = torch.exp(action_dist.distribution.logits)[0].tolist()
#             all_action_probs.append(action_probs)

#         paired_probs = [(action_map[i], round(prob, ndigits=3)) for i, prob in enumerate(action_probs)]
#         print(f"Action Dist: {paired_probs}")
#         env.render()
#         action, _ = model.predict(obs, deterministic=False)
#         print(f"Action: {action_map[int(action)]}")
#         obs, reward, done,_, _ = env.step(int(action))



print(f"Mean Reward: {rew:.2f}, Std Reward: {std:.2f}")