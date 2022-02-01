import sys
import gym
import numpy as np
from envs import grid_spec

from envs.grid_spec import REWARD, REWARD2, REWARD3, REWARD4, WALL, LAVA, TILES, START, RENDER_DICT

ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0,0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1]
}
ACT_TO_STR = {
    ACT_NOOP: 'NOOP',
    ACT_UP: 'UP',
    ACT_LEFT: 'LEFT',
    ACT_RIGHT: 'RIGHT',
    ACT_DOWN: 'DOWN'
}

class TransitionModel(object):
    def __init__(self, gridspec, eps=0.0):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a):
        legal_moves = self.__get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[legal_moves] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0-self.eps
        else:
            p[ACT_NOOP] += 1.0-self.eps
        return p

    def __get_legal_moves(self, s):
        xy = np.array(self.gs.idx_to_xy(s))
        moves = [move for move in ACT_DICT if not self.gs.out_of_bounds(xy+ACT_DICT[move])
                                             and self.gs[xy+ACT_DICT[move]] != WALL]
        return moves

class RewardFunction(object):
    def __init__(self, rew_map=None, default=0.0):
        if rew_map is None:
            rew_map = {
                REWARD: 1.0,
                REWARD2: 2.0,
                REWARD3: 4.0,
                REWARD4: 8.0,
                LAVA: -100.0,
            }
        self.default = default
        self.rew_map = rew_map

    def __call__(self, gridspec, s):
        val = gridspec[gridspec.idx_to_xy(s)]
        if val in self.rew_map:
            return self.rew_map[val]
        return self.default

class GridEnv(gym.Env):
    def __init__(self,
                 grid_spec,
                 max_timesteps=50):

        self.gs = grid_spec
        self.max_timesteps = max_timesteps
        self.num_states = len(grid_spec)
        self.num_actions = 5

        np.random.seed(33)

        # print('self.num_states', self.num_states)
        # print('self.num_actions', self.num_actions)
        # print('gridspec', grid_spec)
        # print('max_timesteps', max_timesteps)

        # Transition model (deterministic actions).
        self.model = TransitionModel(grid_spec, eps=0.0)
        
        # Reward function.
        self.reward = RewardFunction()

        self._timestep = 0
        super(GridEnv, self).__init__()

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.num_actions)

    @property
    def observation_space(self):
        return gym.spaces.Discrete(self.num_states)

    def get_state(self):
        return self.__state

    def set_state(self, s):
        self.__state = s

    def get_transitions(self, s, a):
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA or tile_type == WALL:
            return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(self.num_actions):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.gs.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict

    def get_reward(self, s):
        return self.reward(self.gs, s)

    def step_stateless(self, s, a, verbose=False):
        aprobs = self.model.get_aprobs(s, a)
        samp_a = np.random.choice(range(self.num_actions), p=aprobs)

        next_s = self.gs.idx_to_xy(s) + ACT_DICT[samp_a]
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA:
            next_s = self.gs.idx_to_xy(s)

        next_s_idx = self.gs.xy_to_idx(next_s)
        r = self.reward(self.gs, s)

        return next_s_idx, r

    def step(self, a):
        ns, r = self.step_stateless(self.__state, a)
        self.__state = ns
        done = False
        self._timestep += 1
        if self._timestep >= self.max_timesteps:
            done = True
        return ns, r, done, {}
        
    def reset(self):
        start_idxs = np.array(np.where(self.gs.spec == START)).T
        start_idx = start_idxs[np.random.randint(0, start_idxs.shape[0])]
        start_idx = self.gs.xy_to_idx(start_idx)
        self.__state = start_idx
        self._timestep = 0
        return start_idx

    def render(self, close=False, ostream=sys.stdout):
        print(self.__state)


class GridEnvRandomObservation(GridEnv):

    def __init__(self, obs_dim=8, **kwargs):

        super(GridEnvRandomObservation, self).__init__(**kwargs)

        self.obs_dim = obs_dim
        self._observations = np.random.random((self.num_states, obs_dim)).astype(np.float32)*2 - 1

    @property
    def observation_space(self):
        low = np.array([-1.]*self.obs_dim)
        high = np.array([1.]*self.obs_dim)
        return gym.spaces.Box(low=low, high=high,
                            shape=(self.obs_dim,), dtype=np.float32)
    
    def get_observation(self, state):
        return self._observations[state]

    def step(self, a):
        next_s, r, done, info = super().step(a)
        return self._observations[next_s], r, done, info

    def reset(self):
        state = super().reset()
        return self._observations[state]
