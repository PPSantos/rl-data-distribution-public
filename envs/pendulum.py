"""
    OpenAI gym pendulum environment with discretized actions.
"""

import gym
from gym import spaces
import numpy as np


class PendulumEnv(gym.Env):
    """
        Adapted OpenAI gym pendulum environment.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0, num_actions=5):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.viewer = None

        self.action_map = np.linspace(-self.max_torque, self.max_torque, num=num_actions)

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = self.action_map[u]
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        newth = angle_normalize(newth)

        # Normalize costs to [0,1] range.
        # Check https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        costs = costs / 16.3

        self.state = np.array([newth, newthdot], dtype=np.float32)
        return self.state, -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi