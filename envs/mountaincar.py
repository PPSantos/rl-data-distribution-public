"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MountainCarEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, time_limit=100):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.min_speed = -self.max_speed
        self.goal_position = 0.5

        self.force = 0.001
        self.gravity = 0.0025

        self.time_limit = time_limit
        self.timer = 0

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position, velocity = self.state
        for _ in range(3):
            velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
            velocity = np.clip(velocity, self.min_speed, self.max_speed-1e-8)
            position += velocity
            position = np.clip(position, self.min_position, self.max_position-1e-8)
        if position == self.min_position and velocity < 0:
            velocity = 0

        if position >= self.goal_position:
            reward = 0.0
            #print('Reached goal')
        else:
            reward = -1.0

        self.state = (position, velocity)
        self.timer += 1

        if self.timer >= self.time_limit: # or position >= self.goal_position:
            done = True
        else:
            done = False

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.timer = 0
        return np.array(self.state, dtype=np.float32)

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class DiscreteMountainCarEnv(MountainCarEnv):

    def __init__(self, pos_disc=64, vel_disc=64, **env_args):

        super(DiscreteMountainCarEnv, self).__init__(**env_args)

        self.pos_disc = pos_disc
        self.vel_disc = vel_disc
        self.num_states = pos_disc*vel_disc
        self.num_actions = 3

        self.pos_step = (self.max_position-self.min_position) / self.pos_disc
        self.vel_step = (self.max_speed-self.min_speed) / self.vel_disc

    def get_state(self):
        return self.obs_to_state(self.state)

    def set_state(self, state):
        self.state = self.state_to_obs(state)

    def observation(self, state):
        return self.state_to_obs(state)

    def transitions(self, state, action):
        # Deterministic transition function.
        self.set_state(state)
        self.step(action)
        next_state = self.get_state()
        return {next_state: 1.0} # deterministic transition.

    def reward(self, state, action):
        # Deterministic reward.
        self.set_state(state)
        s_t1, r_t1, done, _ = self.step(action)
        return r_t1

    def obs_to_state(self, obs):
        pos, vel = obs[0], obs[1]
        pos_idx = math.floor((pos-self.min_position)/self.pos_step)
        vel_idx = math.floor((vel-self.min_speed)/self.vel_step)
        return pos_idx + self.pos_disc * vel_idx

    def state_to_obs(self, state):
        pos_idx = state % self.pos_disc
        vel_idx = state // self.pos_disc
        pos = self.min_position + self.pos_step * pos_idx
        vel = self.min_speed + self.vel_step * vel_idx
        pos += 0.5 * self.pos_step
        vel += 0.5 * self.vel_step
        return np.array((pos,vel), dtype=np.float32)
