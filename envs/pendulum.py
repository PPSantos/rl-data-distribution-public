"""
    OpenAI gym pendulum environment with discretized actions.
"""
__credits__ = ["Carlos Luis"]

from multiprocessing.sharedctypes import Value
from typing import Optional

import gym
from gym import spaces
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem is a classic problem in the control literature. In this
    version of the problem, the pendulum starts in a random position, and the goal is to swing it up so
    it stays upright.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta`: angle in radians.
    - `tau`: torque in `N * m`. Defined as positive _counter-clockwise_.

    ## Action Space
    The action is the torque applied to the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards
    The reward is defined as:
    ```
    r = -(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)
    ```
    where `theta` is the pendulum's angle normalized between `[-pi, pi]`.
    Based on the above equation, the minimum reward that can be obtained is `-(pi^2 + 0.1*8^2 +
    0.001*2^2) = -16.2736044`, while the maximum reward is zero (pendulum is
    upright with zero velocity and no torque being applied).

    ## Starting State
    The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.

    ## Episode Termination
    An episode terminates after 200 steps. There's no other criteria for termination.

    ## Arguments
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `g`: acceleration of gravity measured in `(m/s^2)` used to calculate the pendulum dynamics. The default is
    `g=10.0`.

    ```
    gym.make('CartPole-v1', g=9.81)
    ```

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)
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