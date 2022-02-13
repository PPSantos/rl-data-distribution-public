import numpy as np

from gym.envs.classic_control import MountainCarEnv
from gym.wrappers.time_limit import TimeLimit

from envs import grid_env, grid_spec, multipath, pendulum, cartpole
from envs import env_discretizer


GRID_ENVS = {
    'gridEnv1': {
        'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
        'obs_dim': 8,
        'max_timesteps': 50,
    },
    'gridEnv2': {
        'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 7)],
                    grid_spec.WALL: [(7, 7), (5, 1), (2, 4), (5, 6), (6, 0),
                                (0, 4), (3, 4), (3, 7), (2, 1), (7, 0), (4, 7), (5, 5)],
                    grid_spec.REWARD: [(7, 1)]}),
        'obs_dim': 8,
        'max_timesteps': 50,
    },
}


# Environments suite.
ENV_KEYS = ['gridEnv1', 'gridEnv2', 'multiPathEnv', 'mountaincar', 'pendulum', 'cartpole']
def get_env(name):

    if name in ('gridEnv1', 'gridEnv2'):
        env_params = GRID_ENVS[name]
        grid_spec = env_params['grid_spec']
        env = grid_env.GridEnvRandomObservation(grid_spec=grid_spec,
                max_timesteps=env_params['max_timesteps'],
                obs_dim=env_params['obs_dim'])
        return env, grid_spec

    elif name == 'multiPathEnv':
        env = multipath.MultiPathRandomObservation()
        return env, None

    elif name == 'mountaincar':
        env = env_discretizer.get_env(MountainCarEnv)(dim_bins=50)
        env = TimeLimit(env, max_episode_steps=200)
        return env, None

    elif name == 'pendulum':
        env = env_discretizer.get_env(pendulum.PendulumEnv)(dim_bins=50)
        env = TimeLimit(env, max_episode_steps=200)
        return env, None

    elif name == 'cartpole':
        env = env_discretizer.get_env(cartpole.CartPoleEnv)(dim_bins=50)
        env = TimeLimit(env, max_episode_steps=200)
        return env, None

    else:
        raise NotImplementedError('Unknown env id: %s' % name)
