import numpy as np
import gymnasium as gym
from snake_game import Env, SnakeState
import cv2
import itertools
import pygame
import time # Already imported for info['episode']['t']
from collections import defaultdict
from llm_reward_shaper import metrics_collector, get_reward_for_step

# Epsiode length - Removed, now read from params
# MAX_STEPS = 1000

# # Initial game settings - Removed, now read from params
# INIT_HP = 100
# INIT_TAIL_SIZE = 4
# MAX_FRUITS = 1
# PERSPECTIVE = 'third'

# # Rewards - Removed, implementing sparse reward logic
# reward_map = {
#     SnakeState.OK: -0.4,
#     SnakeState.ATE: 20,
#     SnakeState.DED: -40,
#     SnakeState.WON: 1
# }


# Allows rewards to be converted from the json strings to enum values
# Removed parse_enum as reward_map is no longer used directly for step rewards
# def parse_enum(enum, str_dict:dict):
#     return {enum[key.split('.')[-1]]: value for key, value in str_dict.items()}

class SnakeGameEnv(gym.Env):
    """
    Custom Environment for Snake Game using Gymnasium API.
    Reward is calculated based on the dynamically updated global config.
    Episode stats are returned in the info dict upon termination/truncation.
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, max_steps=1000, init_hp=100, init_tail_size=4, num_fruits=1, gs=10, perspective='third', num_snakes=1, num_teams=1, render_mode=None, reward_config=None, rewards=None):
        super().__init__()
        self.env = Env(grid_size=gs, num_fruits=num_fruits, num_snakes=num_snakes, num_teams=num_teams, init_hp=init_hp, init_tail_size=init_tail_size, perspective=perspective)

        if perspective == 'third':
            self.action_map = {
                0: 'up',
                1: 'down',
                2: 'left',
                3: 'right'
            }
            self.action_space = gym.spaces.Discrete(4)
        elif perspective == 'first':
            self.action_map = {
                0: 'stay',
                1: 'left',
                2: 'right'
            }
            self.action_space = gym.spaces.Discrete(3)
        else:
             raise ValueError(f"Invalid perspective: {perspective}. Must be 'first' or 'third'.")

        # self.reward_map = parse_enum(SnakeState, rewards) # Removed reward map usage
        self.max_steps = max_steps
        self.num_snakes = num_snakes
        self.numteams = num_teams
        self.scale = 4 # Scaling factor for rendering observations
        self.render_mode = render_mode
        self.gs = gs # Grid size

        # Store reward configuration from either reward_config or rewards parameter
        if reward_config is not None:
            self.reward_config = reward_config
        elif rewards is not None:
            self.reward_config = rewards
        else:
            raise ValueError("No reward configuration provided")

        # Initialize episode state trackers here
        self._reset_episode_stats()

        # Define observation space (assuming CNN Policy for now)
        # If using MultiInputPolicy, uncomment the Dict space definition
        # n = 5 * self.num_snakes  # Example features per snake
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         'image': gym.spaces.Box(
        #             low=0, high=255, shape=(self.gs*self.scale, self.gs*self.scale, 3),
        #             dtype=np.uint8),
        #         'vector': gym.spaces.Box(
        #             low=-np.inf, high=np.inf, shape=(n,), # Use appropriate bounds
        #             dtype=np.float32) # Use float for vector features
        #     }
        # )

        # FOR CNN Policy
        self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(self.gs*self.scale, self.gs*self.scale, 3),
                    dtype=np.uint8)

        # Setup render window if needed
        self.window = None
        self.clock = None

    def _reset_episode_stats(self):
        """Resets stats tracked within a single episode."""
        self.food_eaten_this_episode = 0
        self.current_episode_length = 0
        self.unique_cells_visited = set()
        self.position_history = [] # Still needed for loop/state checks
        self.actions_this_episode = [] # Track actions for entropy/turns
        self.center_visits_this_episode = 0
        self.turns_this_episode = 0

    def _get_obs(self):
        # Resize image observation
        img_obs = cv2.resize(self.env.to_image(), (self.gs*self.scale, self.gs*self.scale), interpolation=cv2.INTER_NEAREST)

        # # FOR MULTIINPUT Policy (Example vector)
        # # Ensure this matches the feature extractor's expectations
        # snakes_data = []
        # for snake in self.env.snakes:
        #     # Example features: hp, direction (int), head_x, head_y, dist_to_fruit
        #     dist_to_fruit = self.env.get_min_dist_to_fruit() # Assuming single agent perspective for now
        #     snakes_data.extend([snake.hp, snake.direction.to_int(), snake.head.x, snake.head.y, dist_to_fruit])
        # # Pad if necessary (ensure consistent vector size)
        # current_len = len(snakes_data)
        # required_len = 5 * self.num_snakes # Match Dict space definition
        # if current_len < required_len:
        #     snakes_data.extend([0] * (required_len - current_len))
        # vector_obs = np.array(snakes_data, dtype=np.float32)
        # return {'image': img_obs, 'vector': vector_obs}

        # FOR CNN Policy
        return img_obs

    def _get_info(self):
        # Base info, additional stats added on termination
        info = {
            # Add any step-level info if needed, otherwise empty
        }
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self._reset_episode_stats() # Reset internal episode stats

        # Add starting position to history
        head_pos = self.env.snakes[0].head if self.env.snakes else None
        if head_pos:
             pos_tuple = (head_pos.x, head_pos.y)
             self.position_history.append(pos_tuple)
             self.unique_cells_visited.add(pos_tuple)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def is_near_wall(self, head_pos, threshold=1):
        """Check if the snake is near a wall."""
        x, y = head_pos.x, head_pos.y
        return x <= threshold or y <= threshold or x >= self.gs - threshold - 1 or y >= self.gs - threshold - 1
        
    def is_in_center(self, head_pos):
        """Check if the snake is in the center region of the grid."""
        x, y = head_pos.x, head_pos.y
        center_start = self.gs // 3
        center_end = self.gs - center_start
        return center_start <= x < center_end and center_start <= y < center_end
        
    def detect_looping(self, window_size: int = 20, min_loop_length: int = 4) -> bool:
        """Check if the snake is looping in its recent movements."""
        if len(self.position_history) < window_size + min_loop_length:
            return False
            
        recent_positions = self.position_history[-window_size:]
        for pattern_length in range(min_loop_length, window_size // 2 + 1):
            pattern = recent_positions[-pattern_length:]
            previous_segment = recent_positions[-(pattern_length*2):-pattern_length]
            if pattern == previous_segment:
                return True
                
        return False

    def step(self, action):
        # Map action index to game action string
        game_action = self.action_map[action]
        actions = [game_action]

        # Handle actions for other snakes if multi-agent
        if self.num_snakes > 1:
            for _ in range(1, len(self.env.snakes)):
                 actions.append(self.action_map[self.action_space.sample()])

        # Update the game state
        snake_condition, hp, tail_size = self.env.update(actions)
        
        # Update episode metrics
        self.current_episode_length += 1
        
        # Track snake position for metrics
        head_pos = self.env.snakes[0].head if self.env.snakes else None
        is_looping = False
        is_in_center_flag = False
        near_wall_flag = False
        unique_cell_flag = False

        if head_pos:
            pos_tuple = (head_pos.x, head_pos.y)
            self.position_history.append(pos_tuple)
            unique_cell_flag = pos_tuple not in self.unique_cells_visited
            self.unique_cells_visited.add(pos_tuple)

            self.actions_this_episode.append(action)
            if len(self.actions_this_episode) >= 2 and self.actions_this_episode[-1] != self.actions_this_episode[-2]:
                self.turns_this_episode += 1

            if self.is_in_center(head_pos):
                self.center_visits_this_episode += 1
                is_in_center_flag = True

            is_looping = self.detect_looping()
            near_wall_flag = self.is_near_wall(head_pos)

        # Check if food was eaten in this step
        if snake_condition == SnakeState.ATE:
            self.food_eaten_this_episode += 1

        # Determine termination conditions
        terminated = snake_condition in [SnakeState.DED, SnakeState.WON]
        truncated = self.env.time_steps >= self.max_steps

        # Calculate reward using the local config
        reward = get_reward_for_step(
            self.reward_config,
            snake_condition,
            is_looping=is_looping,
            in_center=is_in_center_flag,
            near_wall=near_wall_flag,
            unique_cell=unique_cell_flag
        )
        
        # Prepare info dict
        info = self._get_info()

        # Check if episode is ending
        if terminated or truncated:
            # Determine death cause
            death_cause = "timeout"
            if terminated:
                if snake_condition == SnakeState.DED:
                    if head_pos and not (0 <= head_pos.x < self.gs and 0 <= head_pos.y < self.gs):
                        death_cause = "wall"
                    else:
                        death_cause = "self"
                elif snake_condition == SnakeState.WON:
                    death_cause = "won"

            # Calculate episode action entropy
            action_entropy = 0.0
            if self.actions_this_episode:
                action_counts = defaultdict(int)
                for act in self.actions_this_episode:
                    action_counts[act] += 1
                total_actions = len(self.actions_this_episode)
                probs = [count / total_actions for count in action_counts.values()]
                action_entropy = -sum(p * np.log2(p) for p in probs if p > 0)

            info['episode_stats'] = {
                "length": self.current_episode_length,
                "food_eaten": self.food_eaten_this_episode,
                "death_cause": death_cause,
                "max_length": tail_size + 1,
                "map_coverage": len(self.unique_cells_visited) / (self.gs * self.gs),
                "center_visits": self.center_visits_this_episode,
                "turns": self.turns_this_episode,
                "looping": is_looping,
                "action_entropy": action_entropy
            }
            info["episode"] = {
                "r": reward, # Log final step reward? Or cumulative? SB3 usually handles cumulative.
                "l": self.current_episode_length,
                "t": time.time() # Monitor wrapper usually adds this
            }

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ansi":
            print(self.env.to_string())
        # Add other modes if needed

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((640, 640))
                pygame.display.set_caption("Snake Game")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # Get image from game engine
            im = self.env.to_image(gradation=True)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Resize and create pygame surface
            surf = pygame.surfarray.make_surface(np.rot90(cv2.resize(im_rgb, (640, 640), interpolation=cv2.INTER_NEAREST)))

            self.window.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            im = self.env.to_image(gradation=True)
            return cv2.resize(im, (640, 640), interpolation=cv2.INTER_NEAREST)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        cv2.destroyAllWindows() # Keep this if using direct cv2.imshow anywhere else

