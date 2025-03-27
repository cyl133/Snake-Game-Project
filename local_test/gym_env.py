import numpy as np
import gymnasium as gym
from snake_game import Env, SnakeState
import cv2
import itertools

# Epsiode length
MAX_STEPS = 50

# Initial game settings
INIT_HP = 10
INIT_TAIL_SIZE = 1
MAX_FRUITS = 50

# Rewards
reward_map = {
    SnakeState.OK: 0,
    SnakeState.ATE: 0,  # No immediate reward for eating
    SnakeState.DED: 0,  # No immediate penalty for dying
    SnakeState.WON: 0   # No immediate reward for winning
}

class SnakeGameEnv(gym.Env):
    def __init__(self, gs=10, num_snakes=1, num_teams=1, render_mode='human'):
        super(SnakeGameEnv, self).__init__()
        self.env = Env(gs, num_fruits=MAX_FRUITS, num_snakes=num_snakes, num_teams=num_teams, init_hp=INIT_HP, init_tail_size=INIT_TAIL_SIZE)
        self.action_map = {
            0: 'stay',
            1: 'left',
            2: 'right'
        }
        self.num_snakes = num_snakes
        self.numteams = num_teams
        self.scale = 1
        self.render_scale = 64
        self.render_mode = render_mode
        self.gs = gs
        self.initial_tail_size = INIT_TAIL_SIZE

        self.action_space = gym.spaces.Discrete(len(self.action_map.keys()))

        n = 5 * self.num_snakes  # 5 features per snake, add more if needed
        self.observation_space = gym.spaces.Dict(
            {
                'image': gym.spaces.Box(
                    low=0, high=255, shape=(self.gs*self.scale, self.gs*self.scale, 3),
                    dtype=np.uint8),
                'vector': gym.spaces.Box(
                    low=0, high=100, shape=(n,),
                    dtype=np.int16)
            }
        ) 

    def _get_obs(self):
        snakes = [(snake.hp, snake.direction.to_int(), snake.colour.value, snake.head.x, snake.head.y) for snake in self.env.snakes]
        snakes = list(itertools.chain(*snakes)) + [0, 0, 0] * (self.num_snakes - len(snakes))
        return {
            'image': self.env.to_image(),
            'vector': snakes
        }

    def _get_info(self):
        # If we have snakes available, return relevant metrics
        if hasattr(self.env, 'snakes') and len(self.env.snakes) > 0:
            return {
                'food_eaten': self.env.snakes[0].food_eaten,
                'episode_length': self.env.time_steps
            }
        return {}
        
    def reset(self, *, seed = None):
        super().reset(seed=seed)

        self.env.reset()
        self.initial_tail_size = INIT_TAIL_SIZE
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        actions = [action]
        if self.num_snakes > 1:  #TODO: Add fixed agent move selection, currently random
            for i in range(1, self.num_snakes):
                actions.append(self.action_space.sample())

        snake_condition, hp, tail_size, food_eaten = self.env.update([self.action_map[a] for a in actions])

        is_terminal = snake_condition in [SnakeState.DED, SnakeState.WON] 
        truncated = self.env.time_steps > MAX_STEPS
        
        # Only give reward at the end of the episode based on final snake length
        reward = 0
        if is_terminal or truncated:
            reward = food_eaten  # Scale the reward appropriately
        
        return self._get_obs(), reward, is_terminal, truncated, self._get_info()
    
    def render(self):
        if self.render_mode == 'human':
            im = self.env.to_image()
            cv2.imshow('Snake Game', cv2.resize(im, (self.gs*self.render_scale, self.gs*self.render_scale), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)
        elif self.render_mode == 'rgb_array':
            return cv2.resize(self.env.to_image(), (640, 640), interpolation=cv2.INTER_NEAREST)
        elif self.render_mode == 'ansi':
            print(self.env.to_string())

    def close(self):
        cv2.destroyAllWindows()

