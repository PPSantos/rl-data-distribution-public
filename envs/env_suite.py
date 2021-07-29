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

CUSTOM_PENDULUM_ENVS = {
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

def get_custom_grid_env(env_name, env_type='default', dim_obs=8, time_limit=50, tabular=False,
                        smooth_obs=False, one_hot_obs=False, absorb=False, seed=None):

    if env_name not in CUSTOM_GRID_ENVS.keys():
        raise KeyError('Unknown env. name.')

    if env_type not in CUSTOM_GRID_ENVS[env_name].keys():
        raise KeyError('Unknown env. type.')

    env_params = CUSTOM_GRID_ENVS[env_name][env_type]

    with math_utils.np_seed(seed):
        env = grid_env_cy.GridEnv(env_params['grid_spec'], phi=env_params['phi'])

        if absorb:
            env = env_wrapper.AbsorbingStateWrapper(env)

        if tabular:
            env = wrap_time(env, time_limit=time_limit)
        else:
            env = wrap_obs_time(env, time_limit=time_limit, one_hot_obs=one_hot_obs,
                                dim_obs=dim_obs, smooth_obs=smooth_obs)

    return env, env_params['grid_spec']


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

def get_pendulum_env(env_type='default'):

    if env_type not in CUSTOM_PENDULUM_ENVS.keys():
        raise KeyError('Unknown pendulum env. type.')

    env_params = CUSTOM_PENDULUM_ENVS[env_type]

    env = tabular_env.InvertedPendulum(state_discretization=32,
                                       action_discretization=5,
                                       gravity=env_params['gravity'],
                                       initial_states=env_params['initial_states'],
    )
    env = wrap_time(env, time_limit=50)
    return env

# suite
ENV_KEYS = ['grid16randomobs', 'grid16onehot', 'grid64randomobs', 'grid64onehot', 'cliffwalk', 'pendulum', 'mountaincar', 'sparsegraph']
def get_env(name):
    if name == 'grid16randomobs':
        env = random_grid_env(16, 16, dim_obs=16, time_limit=50, wall_ratio=0.2, smooth_obs=False, seed=0)
    elif name == 'grid16onehot':
        env = random_grid_env(16, 16, time_limit=50, wall_ratio=0.2, one_hot_obs=True, seed=0)
    elif name == 'grid16sparse':
        env = random_grid_env(16, 16, time_limit=50, wall_ratio=0.2, one_hot_obs=True, seed=0, distance_reward=False)
    elif name == 'grid64randomobs':
        env = random_grid_env(64, 64, dim_obs=64, time_limit=100, wall_ratio=0.2, smooth_obs=False, seed=0)
    elif name == 'grid64onehot':
        env = random_grid_env(64, 64, time_limit=100, wall_ratio=0.2, one_hot_obs=True, seed=0)
    elif name == 'cliffwalk':
        with math_utils.np_seed(0):
            env = tabular_env.CliffwalkEnv(25)
            # Cliffwalk is unsolvable by QI with moderate entropy - up the reward to reduce the effects.
            env = env_wrapper.AbsorbingStateWrapper(env, absorb_reward=10.0)
            env = wrap_obs_time(env, dim_obs=16, time_limit=50)
    elif name == 'pendulum':
        env = tabular_env.InvertedPendulum(state_discretization=32, action_discretization=5)
        env = wrap_time(env, time_limit=50)
    elif name == 'mountaincar':
        env = tabular_env.MountainCar(posdisc=56, veldisc=32)
        # MountainCar is unsolvable by QI with moderate entropy - up the reward to reduce the effects.
        env = env_wrapper.AbsorbingStateWrapper(env, absorb_reward=10.0)  
        env = wrap_time(env, time_limit=100)
    elif name == 'sparsegraph':
        with math_utils.np_seed(0):
            env = tabular_env.RandomTabularEnv(num_states=500, num_actions=3, transitions_per_action=1, self_loop=True)
            env = env_wrapper.AbsorbingStateWrapper(env, absorb_reward=10.0)
            env = wrap_obs_time(env, dim_obs=4, time_limit=10)
    elif name == 'multiPathsEnv':
        env = tabular_env.MultiPathsEnv()
        # env = random_obs_wrapper.MultiPathsEnvObsWrapper(env, dim_obs=5)
        env = random_obs_wrapper.MultiPathsEnvObsWrapper1Hot(env)
        env = time_limit_wrapper.TimeLimitWrapper(env, time_limit=10)
    elif name == 'mdp1':
        env = tabular_env.MDP1()
        env = time_limit_wrapper.TimeLimitWrapper(env, time_limit=5)
    else:
        raise NotImplementedError('Unknown env id: %s' % name)
    return env
