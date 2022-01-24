import numpy as np

from envs import grid_env, grid_spec
from envs import mountaincar, multipath

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
ENV_KEYS = ['gridEnv1', 'gridEnv2', 'multiPathEnv'] # 'pendulum', 'mountaincar', 'mdp1']
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

    elif name == 'pendulum':
        raise ValueError('Env. not implemented.')
        # Load default env.
        default_params = PENDULUM_ENVS['default']
        train_env = tabular_env.InvertedPendulum(state_discretization=32,
                                                 action_discretization=5,
                                                 gravity=default_params['gravity'],
                                                 initial_states=default_params['initial_states'])
        train_env = wrap_time(train_env, time_limit=50)

        # Load rollouts envs.
        rollouts_envs = []
        for r_type, r_env_params in sorted(PENDULUM_ENVS.items()):
            r_env = tabular_env.InvertedPendulum(state_discretization=32,
                                                 action_discretization=5,
                                                 gravity=r_env_params['gravity'],
                                                 initial_states=r_env_params['initial_states'])
            r_env = wrap_time(r_env, time_limit=50)
            rollouts_envs.append(r_env)

        env_grid_spec = None
        return train_env, env_grid_spec, rollouts_envs

    elif name == 'mountaincar':
        raise ValueError('Env. not implemented.')
        # Load default env.
        default_params = MOUNTAINCAR_ENVS['default']
        train_env = mountaincar.DiscreteMountainCarEnv(pos_disc=100, vel_disc=100, time_limit=200)

        # Load rollouts envs.
        rollouts_envs = []
        for r_type, r_env_params in sorted(MOUNTAINCAR_ENVS.items()):
            r_env = mountaincar.DiscreteMountainCarEnv(pos_disc=100, vel_disc=100, time_limit=200)
            rollouts_envs.append(r_env)

        env_grid_spec = None
        return train_env, env_grid_spec, rollouts_envs

    elif name == 'mdp1':
        raise ValueError('Env. not implemented.')
        env = tabular_env.MDP1()
        env = time_limit_wrapper.TimeLimitWrapper(env, time_limit=5)

        r_env = tabular_env.MDP1()
        r_env = time_limit_wrapper.TimeLimitWrapper(r_env, time_limit=5)

        env_grid_spec = None
        return env, env_grid_spec, [r_env]

    else:
        raise NotImplementedError('Unknown env id: %s' % name)
