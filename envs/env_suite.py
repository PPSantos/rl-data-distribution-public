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


CUSTOM_ENVS = {
    'gridEnv1': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 7)],
                    TileType.WALL: [],
                    TileType.REWARD: [(7, 0)]}),

    'gridEnv2': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 7)],
                    TileType.WALL: [(7, 7), (5, 1), (2, 4), (5, 6), (6, 0),
                                (0, 4), (3, 4), (3, 7), (2, 1), (7, 0), (4, 7), (5, 5)],
                    TileType.REWARD: [(7, 1)]}),

    'gridEnv3': grid_spec_cy.spec_from_sparse_locations(8, 8,
                    {TileType.START: [(0, 7)],
                    TileType.WALL: [(5, 2), (5, 0), (7, 6), (2, 1), (7, 0),
                                (7, 1), (4, 0), (5, 3), (7, 2), (2, 2),
                                (2, 5), (4, 2), (3, 5), (3, 3), (4, 7), (2, 0)],
                    TileType.REWARD: [(3, 0)]}),
}

def get_custom_env(env_name, dim_obs=8, time_limit=50, tabular=False,
    smooth_obs=False, one_hot_obs=False, absorb=False):

    if env_name not in CUSTOM_ENVS.keys():
        raise KeyError('Unknown env. name.')
    
    gs = CUSTOM_ENVS[env_name]
    env = grid_env_cy.GridEnv(gs)

    if absorb:
        env = env_wrapper.AbsorbingStateWrapper(env)

    if tabular:
        env = wrap_time(env, time_limit=time_limit)
    else:
        env = wrap_obs_time(env, time_limit=time_limit, one_hot_obs=one_hot_obs,
                            dim_obs=dim_obs, smooth_obs=smooth_obs)
    return env, gs


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
    else:
        raise NotImplementedError('Unknown env id: %s' % name)
    return env
