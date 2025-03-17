import cv2
import numpy as np
from gym_env import SnakeGameEnv

env = SnakeGameEnv(num_snakes=1, num_teams=1)

obs = env.reset()

print(obs)

done = False
while not done:
    # Random action for testing
    action = env.action_space.sample()
    print(f"Action: {action}")

    # Apply the action
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"New observation: {obs['vector']}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

    env.render()