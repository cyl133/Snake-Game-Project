import numpy as np
import gymnasium as gym
from snake_game import Env, SnakeState
import cv2
import itertools
import pygame

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
    Reward is sparse: given only at the end of the episode,
    equal to the total number of fruits eaten.
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    # Updated __init__ signature for clarity and removed unused rewards param from here
    def __init__(self, max_steps=1000, init_hp=100, init_tail_size=4, num_fruits=1, gs=10, perspective='third', num_snakes=1, num_teams=1, render_mode=None):
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

        # Initialize episode-specific counters
        self.food_eaten_this_episode = 0

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
        # Returns info dictionary, populated at the end of the episode in step()
        # Can add other persistent info if needed, e.g., distance to fruit
        # return {"distance_to_fruit": self.env.get_min_dist_to_fruit()}
        return {} # Keep simple for now, essential info added in step()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset()
        self.food_eaten_this_episode = 0 # Reset food counter

        observation = self._get_obs()
        info = self._get_info() # Initial info

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map action index to game action string if needed
        game_action = self.action_map[action]
        actions = [game_action] # Assume first action is for the agent being trained

        # Handle actions for other snakes if multi-agent (currently random)
        if self.num_snakes > 1:
            # TODO: Implement proper multi-agent action handling if needed
            for _ in range(1, len(self.env.snakes)): # Use current number of snakes
                 actions.append(self.action_map[self.action_space.sample()])

        # Update the game state
        snake_condition, hp, tail_size = self.env.update(actions)

        # Check if food was eaten in this step
        if snake_condition == SnakeState.ATE:
            self.food_eaten_this_episode += 1

        # Determine termination conditions
        terminated = snake_condition in [SnakeState.DED, SnakeState.WON]
        truncated = self.env.time_steps >= self.max_steps # Use >= for clarity

        # Calculate reward (only at the end of the episode)
        reward = 0.0 # No reward during the episode
        info = {} # Reset info for this step

        if terminated or truncated:
            reward = float(self.food_eaten_this_episode) # Final reward = total food eaten
            # Populate info dictionary for logging/callbacks
            info["food_eaten"] = self.food_eaten_this_episode
            info["episode_length"] = self.env.time_steps
            # Include the terminal observation? SB3 handles this generally.
            # info["terminal_observation"] = self._get_obs()

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        # Return according to Gymnasium API: obs, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame() # Frame rendering handled internally now
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
            im = self.env.to_image(gradation=True) # BGR format from cv2
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Convert to RGB for pygame

            # Resize and create pygame surface
            surf = pygame.surfarray.make_surface(np.rot90(cv2.resize(im_rgb, (640, 640), interpolation=cv2.INTER_NEAREST)))

            self.window.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            im = self.env.to_image(gradation=True)
            return cv2.resize(im, (640, 640), interpolation=cv2.INTER_NEAREST) # Return numpy array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        cv2.destroyAllWindows() # Keep this if using direct cv2.imshow anywhere else

