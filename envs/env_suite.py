import numpy as np

from envs import grid_env, grid_spec
from envs import mountaincar

GRID_ENVS = {
    'gridEnv1': {
        'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
        'obs_dim': 8,
        'max_timesteps': 50,
    }
}

# WARNING: Custom grid envs must be square.
CUSTOM_GRID_ENVS = {

    'gridEnv1': {
        'default': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.0,
        },
        'stochastic_actions_1': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.1,
        },
        'stochastic_actions_2': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.2,
        },
        'stochastic_actions_3': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.3,
        },
        'stochastic_actions_4': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.4,
        },
        'stochastic_actions_5': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.5,
        },
        'uniform_init_state_dist': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                     {grid_spec.START: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
                     grid_spec.WALL: [],
                     grid_spec.REWARD: [(7, 0)]}),
            'phi': 0.0,
        }
    },

    'gridEnv2': {
        'default': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 7)],
                    grid_spec.WALL: [(7, 7), (5, 1), (2, 4), (5, 6), (6, 0),
                                (0, 4), (3, 4), (3, 7), (2, 1), (7, 0), (4, 7), (5, 5)],
                    grid_spec.REWARD: [(7, 1)]}),
            'phi': 0.0,
        }
    },

    'gridEnv3': {
        'default': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [(5, 2), (5, 0), (7, 6), (2, 1), (7, 0),
                                        (7, 1), (4, 0), (5, 3), (7, 2), (2, 2),
                                        (2, 5), (4, 2), (3, 5), (3, 3), (4, 7), (2, 0)],
                            grid_spec.REWARD: [(3, 0)]}),
            'phi': 0.0,
        }
    },

    'gridEnv4': {
        'default': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.0,
        },
        'stochastic_actions_1': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.1,
        },
        'stochastic_actions_2': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.2,
        },
        'stochastic_actions_3': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.3,
        },
            'stochastic_actions_4': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.4,
        },
            'stochastic_actions_5': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.5,
        },
        'uniform_init_state_dist': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                     {grid_spec.START: [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
                     grid_spec.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                 (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                     grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.0,
        }
    },  

    'gridEnv5': {
        'default': {
            'grid_spec': grid_spec.spec_from_sparse_locations(5, 5,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(1, 4), (2, 4), (3, 4)],
                    grid_spec.REWARD: [(4, 4)]}),
            'phi': 0.0,
        }
    },

    'lavaEnv1': {
        'default': {
            'grid_spec': grid_spec.spec_from_sparse_locations(8, 8,
                    {grid_spec.START: [(0, 4)],
                    grid_spec.WALL: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
                                (2, 1), (3, 0), (3, 1), (4, 0), (4, 1),
                                (5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1),
                                (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    grid_spec.LAVA: [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                                (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6)],
                    grid_spec.REWARD: [(7, 4)]}),
            'phi': 0.0,
        }
    },

}

PENDULUM_ENVS = {
    'default': {
        'gravity': 10.0,
        'initial_states': [-np.pi/4],
    },
    'uniform_init_state_dist': {
        'gravity': 10.0,
        'initial_states': [-np.pi, -(3/4)*np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, (3/4)*np.pi],
    },
    'gravity_4': {
        'gravity': 4.0,
        'initial_states': [-np.pi/4],
    },
    'gravity_6': {
        'gravity': 6.0,
        'initial_states': [-np.pi/4],
    },
    'gravity_8': {
        'gravity': 8.0,
        'initial_states': [-np.pi/4],
    },
    'gravity_12': {
        'gravity': 12.0,
        'initial_states': [-np.pi/4],
    },
    'gravity_14': {
        'gravity': 14.0,
        'initial_states': [-np.pi/4],
    },
    'gravity_16': {
        'gravity': 16.0,
        'initial_states': [-np.pi/4],
    },
}

MOUNTAINCAR_ENVS = {
    'default': {
        # 'gravity': 0.0025,
        # 'initial_states': [-0.5],
    },
    # 'uniform_init_state_dist': {
    #     'gravity': 0.0025,
    #     'initial_states': [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4,
    #                        -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
    # },
    # 'gravity_0020': {
    #     'gravity': 0.0020,
    #     'initial_states': [-0.5],
    # },
    # 'gravity_0023': {
    #     'gravity': 0.0023,
    #     'initial_states': [-0.5],
    # },
    # 'gravity_0027': {
    #     'gravity': 0.0027,
    #     'initial_states': [-0.5],
    # },
    # 'gravity_0030': {
    #     'gravity': 0.0030,
    #     'initial_states': [-0.5],
    # },
}

MULTIPATHS_ENVS = {
    'default': {
        'initial_states': [0],
        'init_action_random_p': 0.01,
    },
    'uniform_init_state_dist': {
        'initial_states': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        'init_action_random_p': 0.01,
    },
    'action_random_p_0.05': {
        'initial_states': [0],
        'init_action_random_p': 0.05,
    },
    'action_random_p_0.1': {
        'initial_states': [0],
        'init_action_random_p': 0.1,
    },
    'action_random_p_0.2': {
        'initial_states': [0],
        'init_action_random_p': 0.2,
    },
}

# Environments suite.
ENV_KEYS = ['gridEnv1', 'gridEnv4', 'pendulum', 'mountaincar', 'multiPathsEnv', 'mdp1']
def get_env(name, seed=None):

    if name == 'gridEnv1':
        env_params = GRID_ENVS[name]
        grid_spec = env_params['grid_spec']
        env = grid_env.GridEnvRandomObservation(grid_spec=grid_spec,
                max_timesteps=env_params['max_timesteps'],
                obs_dim=env_params['obs_dim'], seed=seed)
        return env, grid_spec

    elif name == 'gridEnv4':
        raise ValueError('Env. not implemented.')
        env, env_grid_spec, rollouts_envs = get_custom_grid_env(env_name=name, dim_obs=8,
                                                        time_limit=50, tabular=False,
                                                        smooth_obs=False, one_hot_obs=False,
                                                        absorb=False, seed=seed)
        return env, env_grid_spec, rollouts_envs

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

    elif name == 'multiPathsEnv':
        raise ValueError('Env. not implemented.')
        # Load default env.
        default_params = MULTIPATHS_ENVS['default']
        with math_utils.np_seed(seed):
            train_env = tabular_env.MultiPathsEnv(init_action_random_p=default_params['init_action_random_p'],
                                            initial_states=default_params['initial_states'])
            train_env = random_obs_wrapper.MultiPathsEnvObsWrapper(train_env, dim_obs=4)
            #train_env = random_obs_wrapper.MultiPathsEnvObsWrapper1Hot(train_env)
            train_env = time_limit_wrapper.TimeLimitWrapper(train_env, time_limit=10)

        # Load rollouts envs.
        rollouts_envs = []
        for r_type, r_env_params in sorted(MULTIPATHS_ENVS.items()):
            with math_utils.np_seed(seed):
                r_env = tabular_env.MultiPathsEnv(init_action_random_p=r_env_params['init_action_random_p'],
                                            initial_states=r_env_params['initial_states'])
                r_env = random_obs_wrapper.MultiPathsEnvObsWrapper(r_env, dim_obs=4)
                #r_env = random_obs_wrapper.MultiPathsEnvObsWrapper1Hot(r_env)
                r_env = time_limit_wrapper.TimeLimitWrapper(r_env, time_limit=10)            
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
