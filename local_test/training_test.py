import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SnakeGameEnv

# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
#         super().__init__(observation_space)

#         n_input_channels = observation_space['image'].shape[-1]
#         print(n_input_channels)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()['image']).float()
#             ).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#         # snake_input_size = observation_space['snakes'].shape[0]
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(snake_input_size, 8),
#         #     nn.ReLU()
#         # )

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         im = observations['image']
#         # snakes = observations['snakes']

#         image_features = self.cnn(im)
#         # image_features = self.linear(image_features)

#         # snake_features = self.mlp(snakes)

#         # combined_features = th.cat([image_features, snake_features], dim=1)
#         return image_features

# policy_kwargs = dict(
#     features_extractor_class=CustomCombinedExtractor,
#     features_extractor_kwargs=dict(features_dim=256), 
# )

# class CustomCNNPolicy(PPO.policy_class):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs, features_extractor_class=CustomCNN)

env = SnakeGameEnv(num_snakes=1, num_teams=1)

model = PPO('MultiInputPolicy', env, verbose=True, device='cuda', n_steps=4)
# model = PPO.load("ppo_snake", env=env, device="cuda")

model.learn(8, progress_bar=1)
model.save('ppo_snake')