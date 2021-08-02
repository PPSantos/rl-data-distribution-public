import numpy as np
import itertools
import random
from envs import random_obs_wrapper, time_limit_wrapper, env_wrapper
from rlutil.envs.tabular_cy import tabular_env
from rlutil.envs.gridcraft import grid_env_cy
from rlutil.envs.gridcraft import grid_spec_cy
from rlutil.logging import log_utils
from rlutil import math_utils
from rlutil.envs.gridcraft.grid_spec_cy import TileType


# WARNING: Custom grid envs must be square.
CUSTOM_GRID_ENVS = {

    'gridEnv1': {
        'default': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [],
                            TileType.REWARD: [(7, 0)]}),
            'phi': 0.0,
        },
        'stochastic_actions_1': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [],
                            TileType.REWARD: [(7, 0)]}),
            'phi': 0.1,
        },
        'stochastic_actions_2': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [],
                            TileType.REWARD: [(7, 0)]}),
            'phi': 0.2,
        },
        'stochastic_actions_3': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [],
                            TileType.REWARD: [(7, 0)]}),
            'phi': 0.3,
        },
        'stochastic_actions_4': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [],
                            TileType.REWARD: [(7, 0)]}),
            'phi': 0.4,
        },
        'stochastic_actions_5': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [],
                            TileType.REWARD: [(7, 0)]}),
            'phi': 0.5,
        },
        'uniform_init_state_dist': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                     {TileType.START: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
                     TileType.WALL: [],
                     TileType.REWARD: [(7, 0)]}),
            'phi': 0.0,
        }
    },

    'gridEnv2': {
        'default': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 7)],
                    TileType.WALL: [(7, 7), (5, 1), (2, 4), (5, 6), (6, 0),
                                (0, 4), (3, 4), (3, 7), (2, 1), (7, 0), (4, 7), (5, 5)],
                    TileType.REWARD: [(7, 1)]}),
            'phi': 0.0,
        }
    },

    'gridEnv3': {
        'default': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                            {TileType.START: [(0, 7)],
                            TileType.WALL: [(5, 2), (5, 0), (7, 6), (2, 1), (7, 0),
                                        (7, 1), (4, 0), (5, 3), (7, 2), (2, 2),
                                        (2, 5), (4, 2), (3, 5), (3, 3), (4, 7), (2, 0)],
                            TileType.REWARD: [(3, 0)]}),
            'phi': 0.0,
        }
    },

    'gridEnv4': {
        'default': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.REWARD: [(7, 4)]}),
            'phi': 0.0,
        },
        'stochastic_actions_1': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.REWARD: [(7, 4)]}),
            'phi': 0.1,
        },
        'stochastic_actions_2': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.REWARD: [(7, 4)]}),
            'phi': 0.2,
        },
        'stochastic_actions_3': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.REWARD: [(7, 4)]}),
            'phi': 0.3,
        },
            'stochastic_actions_4': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.REWARD: [(7, 4)]}),
            'phi': 0.4,
        },
            'stochastic_actions_5': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.REWARD: [(7, 4)]}),
            'phi': 0.5,
        },
        'uniform_init_state_dist': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                     {TileType.START: [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
                     TileType.WALL: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
                                 (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                     TileType.REWARD: [(7, 4)]}),
            'phi': 0.0,
        }
    },  

    'gridEnv5': {
        'default': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(5, 5,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(1, 4), (2, 4), (3, 4)],
                    TileType.REWARD: [(4, 4)]}),
            'phi': 0.0,
        }
    },

    'lavaEnv1': {
        'default': {
            'grid_spec': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 4)],
                    TileType.WALL: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
                                (2, 1), (3, 0), (3, 1), (4, 0), (4, 1),
                                (5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1),
                                (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
                                (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
                    TileType.LAVA: [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                                (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6)],
                    TileType.REWARD: [(7, 4)]}),
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
        'gravity': 0.0025,
        'initial_states': [-0.5],
    },
    'uniform_init_state_dist': {
        'gravity': 0.0025,
        'initial_states': [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4,
                           -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
    },
    'gravity_0020': {
        'gravity': 0.0020,
        'initial_states': [-0.5],
    },
    'gravity_0023': {
        'gravity': 0.0023,
        'initial_states': [-0.5],
    },
    'gravity_0027': {
        'gravity': 0.0027,
        'initial_states': [-0.5],
    },
    'gravity_0030': {
        'gravity': 0.0030,
        'initial_states': [-0.5],
    },
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


def get_custom_grid_env(env_name, dim_obs=8, time_limit=50, tabular=False, smooth_obs=False,
                        one_hot_obs=False, absorb=False, seed=None):

    if env_name not in CUSTOM_GRID_ENVS.keys():
        raise KeyError('Unknown env. name.')

    # Load default env.
    default_env_params = CUSTOM_GRID_ENVS[env_name]['default']
    with math_utils.np_seed(seed):
        train_env = grid_env_cy.GridEnv(default_env_params['grid_spec'], phi=default_env_params['phi'])
        if absorb:
            train_env = env_wrapper.AbsorbingStateWrapper(train_env)
        if tabular:
            train_env = wrap_time(train_env, time_limit=time_limit)
        else:
            train_env = wrap_obs_time(train_env, time_limit=time_limit, one_hot_obs=one_hot_obs,
                                dim_obs=dim_obs, smooth_obs=smooth_obs)

    # Load rollouts envs.
    rollouts_envs = []
    for r_type, r_env_params in sorted(CUSTOM_GRID_ENVS[env_name].items()):
        with math_utils.np_seed(seed):
            r_env = grid_env_cy.GridEnv(r_env_params['grid_spec'], phi=r_env_params['phi'])
            if absorb:
                r_env = env_wrapper.AbsorbingStateWrapper(r_env)
            if tabular:
                r_env = wrap_time(r_env, time_limit=time_limit)
            else:
                r_env = wrap_obs_time(r_env, time_limit=time_limit, one_hot_obs=one_hot_obs,
                                    dim_obs=dim_obs, smooth_obs=smooth_obs)
        rollouts_envs.append(r_env)

    return train_env, default_env_params['grid_spec'], rollouts_envs

def random_grid_env(size_x, size_y, dim_obs=32, time_limit=50,
    wall_ratio=0.1, smooth_obs=False, distance_reward=True,
    one_hot_obs=False, seed=None, absorb=False, tabular=False):
    total_size = size_x * size_y
    locations = list(itertools.product(range(size_x), range(size_y)))
    #start_loc = (int(size_x/2), int(size_y/2))
    start_loc = (0, int(size_y)-1) # start at bottom left.
    locations.remove(start_loc)

    with math_utils.np_seed(seed):
        # randomly place walls
        wall_locs = random.sample(locations, int(total_size*wall_ratio))
        [locations.remove(loc) for loc in wall_locs]

        cand_reward_locs = random.sample(locations, int(0.25 * total_size))
        # pick furthest one from center
        cand_reward_dists = [np.linalg.norm(np.array(reward_loc) - start_loc) for reward_loc in cand_reward_locs]
        furthest_reward = np.argmax(cand_reward_dists)
        reward_loc = cand_reward_locs[furthest_reward]
        locations.remove(cand_reward_locs[furthest_reward])

        # print(start_loc)
        # print(wall_locs)
        # print(reward_loc)

        gs = grid_spec_cy.spec_from_sparse_locations(size_x, size_y, {TileType.START: [start_loc],
                                                            TileType.WALL: wall_locs,
                                                            TileType.REWARD: [reward_loc]})

        if distance_reward:
            env = grid_env_cy.DistanceRewardGridEnv(gs, reward_loc[0], reward_loc[1], start_loc[0], start_loc[1])
        else:
            env = grid_env_cy.GridEnv(gs)

        # Something is wrong here. It seems that while the transition_matrix variable is being
        # modified, the simulation is actually not taking this into consideration.
        # env = env_wrapper.StochasticActionWrapper(env, eps=0.05) 

        if absorb:
            env = env_wrapper.AbsorbingStateWrapper(env)

        if tabular:
            env = wrap_time(env, time_limit=time_limit)
        else:
            env = wrap_obs_time(env, time_limit=time_limit, one_hot_obs=one_hot_obs, dim_obs=dim_obs, smooth_obs=smooth_obs)

    return env, gs

def wrap_obs_time(env, dim_obs=32, time_limit=50, smooth_obs=False, one_hot_obs=False):
    if smooth_obs:
        env = random_obs_wrapper.LocalObsWrapper(env, dim_obs=dim_obs)
    elif one_hot_obs:
        env = random_obs_wrapper.OneHotObsWrapper(env)
    else:
        env = random_obs_wrapper.RandomObsWrapper(env, dim_obs=dim_obs)
    env = time_limit_wrapper.TimeLimitWrapper(env, time_limit=time_limit)
    return env

def wrap_time(env, time_limit=50):
    return time_limit_wrapper.TimeLimitWrapper(env, time_limit=time_limit)


# Environments suite.
ENV_KEYS = ['pendulum', 'mountaincar', 'multiPathsEnv', 'mdp1']
def get_env(name, seed):

    if name == 'pendulum':
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

        return train_env, rollouts_envs

    elif name == 'mountaincar':
        # Load default env.
        default_params = MOUNTAINCAR_ENVS['default']
        train_env = tabular_env.MountainCar(posdisc=56, veldisc=32,
                                            gravity=default_params['gravity'],
                                            initial_states=default_params['initial_states'])
        train_env = env_wrapper.AbsorbingStateWrapper(train_env, absorb_reward=10.0)  
        train_env = wrap_time(train_env, time_limit=100)

        # Load rollouts envs.
        rollouts_envs = []
        for r_type, r_env_params in sorted(MOUNTAINCAR_ENVS.items()):
            r_env = tabular_env.MountainCar(posdisc=56, veldisc=32,
                                            gravity=r_env_params['gravity'],
                                            initial_states=r_env_params['initial_states'])
            r_env = env_wrapper.AbsorbingStateWrapper(r_env, absorb_reward=10.0)  
            r_env = wrap_time(r_env, time_limit=100)
            rollouts_envs.append(r_env)

        return train_env, rollouts_envs

    elif name == 'multiPathsEnv':
        # Load default env.
        default_params = MULTIPATHS_ENVS['default']
        with math_utils.np_seed(seed):
            train_env = tabular_env.MultiPathsEnv(init_action_random_p=default_params['init_action_random_p'],
                                            initial_states=default_params['initial_states'])
            train_env = random_obs_wrapper.MultiPathsEnvObsWrapper(train_env, dim_obs=8)
            #train_env = random_obs_wrapper.MultiPathsEnvObsWrapper1Hot(train_env)
            train_env = time_limit_wrapper.TimeLimitWrapper(train_env, time_limit=10)

        # Load rollouts envs.
        rollouts_envs = []
        for r_type, r_env_params in sorted(MULTIPATHS_ENVS.items()):
            with math_utils.np_seed(seed):
                r_env = tabular_env.MultiPathsEnv(init_action_random_p=r_env_params['init_action_random_p'],
                                            initial_states=r_env_params['initial_states'])
                r_env = random_obs_wrapper.MultiPathsEnvObsWrapper(r_env, dim_obs=8)
                #r_env = random_obs_wrapper.MultiPathsEnvObsWrapper1Hot(r_env)
                r_env = time_limit_wrapper.TimeLimitWrapper(r_env, time_limit=10)            
            rollouts_envs.append(r_env)

        return train_env, rollouts_envs

    elif name == 'mdp1':
        env = tabular_env.MDP1()
        env = time_limit_wrapper.TimeLimitWrapper(env, time_limit=5)
        return env, []

    else:
        raise NotImplementedError('Unknown env id: %s' % name)

    # if name == 'grid16randomobs':
    #     env = random_grid_env(16, 16, dim_obs=16, time_limit=50, wall_ratio=0.2, smooth_obs=False, seed=0)
    # elif name == 'grid16onehot':
    #     env = random_grid_env(16, 16, time_limit=50, wall_ratio=0.2, one_hot_obs=True, seed=0)
    # elif name == 'grid16sparse':
    #     env = random_grid_env(16, 16, time_limit=50, wall_ratio=0.2, one_hot_obs=True, seed=0, distance_reward=False)
    # elif name == 'grid64randomobs':
    #     env = random_grid_env(64, 64, dim_obs=64, time_limit=100, wall_ratio=0.2, smooth_obs=False, seed=0)
    # elif name == 'grid64onehot':
    #     env = random_grid_env(64, 64, time_limit=100, wall_ratio=0.2, one_hot_obs=True, seed=0)
    # elif name == 'cliffwalk':
    #     with math_utils.np_seed(0):
    #         env = tabular_env.CliffwalkEnv(25)
    #         # Cliffwalk is unsolvable by QI with moderate entropy - up the reward to reduce the effects.
    #         env = env_wrapper.AbsorbingStateWrapper(env, absorb_reward=10.0)
    #         env = wrap_obs_time(env, dim_obs=16, time_limit=50)
    # elif name == 'sparsegraph':
    #     with math_utils.np_seed(0):
    #         env = tabular_env.RandomTabularEnv(num_states=500, num_actions=3, transitions_per_action=1, self_loop=True)
    #         env = env_wrapper.AbsorbingStateWrapper(env, absorb_reward=10.0)
    #         env = wrap_obs_time(env, dim_obs=4, time_limit=10)
