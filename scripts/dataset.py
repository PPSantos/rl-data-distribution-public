import json
import random
import numpy as np
import scipy.stats

from utils.json_utils import NumpyEncoder
from envs import env_suite, grid_spec

DEFAULT_DATASET_ARGS = {

    # Environment name.
    'env_name': 'gridEnv1',

    # Dataset type.
    # If type=dirichlet: the dataset is sampled from a distribution that
    # is itself sampled from a prior Dirichlet distribution parameterized by alpha.
    'dataset_type': 'dirichlet',

    # Number of dataset transitions.
    'dataset_size': 50_000,

    # Whether to force coverage over all (s,a) pairs, i.e.,
    # the sampling distribution always verifies p(s,a) > 0.
    'force_full_coverage': False,

    # type=dirichlet args.
    'dirichlet_dataset_args': {
        'dirichlet_alpha_coef': 5.0,
    },
}


def _dataset_from_sampling_dist(env, env_grid_spec, sampling_dist: np.ndarray,
                dataset_size: int, force_full_coverage: bool):

    print('Creating dataset.')

    transitions = []

    mesh = np.array(np.meshgrid(np.arange(env.num_states),
                                np.arange(env.num_actions)))
    sa_combinations = mesh.T.reshape(-1, 2)
    sa_counts = np.zeros((env.num_states, env.num_actions))

    for _ in range(dataset_size):

        # Randomly sample (state, action) pair according to sampling dist.
        tile_type = grid_spec.WALL
        while tile_type == grid_spec.WALL:
            sampled_idx = np.random.choice(np.arange(len(sampling_dist)), p=sampling_dist)
            state, action = sa_combinations[sampled_idx]
            xy = env_grid_spec.idx_to_xy(state)
            tile_type = env_grid_spec.get_value(xy, xy=True)

        sa_counts[state, action] += 1
        observation = env.get_observation(state)

        # Sample next state, observation and reward.
        env.set_state(state)
        next_observation, reward, done, info = env.step(action)

        transitions.append((observation, action, reward, next_observation))

        env.reset()

    if force_full_coverage:
        # Correct dataset such that we have coverage over all (state, action) pairs.
        zero_positions = np.where(sa_counts == 0)
        print('Number of missing (s,a) pairs:', np.sum((sa_counts == 0)))
        for (state, action) in zip(*zero_positions):

            # Skip walls.
            xy = env_grid_spec.idx_to_xy(state)
            tile_type = env_grid_spec.get_value(xy, xy=True)
            if tile_type == grid_spec.WALL:
                continue

            observation = env.get_observation(state)

            # Sample next state, observation and reward.
            env.set_state(state)
            next_observation, reward, done, info = env.step(action)

            transitions.append((observation, action, reward, next_observation))

            env.reset()

    # Flag walls with nan values.
    for state in range(env.num_states):
        for action in range(env.num_actions):
            xy = env_grid_spec.idx_to_xy(state)
            tile_type = env_grid_spec.get_value(xy, xy=True)
            if tile_type == grid_spec.WALL:
                sa_counts[state,action] = np.nan

    sa_counts = sa_counts.flatten() # [S*A]
    sa_counts = sa_counts[np.logical_not(np.isnan(sa_counts))] # remove nans
    dataset_dist = sa_counts / np.sum(sa_counts) # [S*A]

    dataset_info = {}
    dataset_info['dataset_dist'] = dataset_dist
    dataset_info['dataset_sa_counts'] = sa_counts
    dataset_info['dataset_entropy'] = scipy.stats.entropy(dataset_dist)

    return transitions, dataset_info


def main(args=None):

    print('\nRunning scripts/dataset.py.')

    if not args:
        args = DEFAULT_DATASET_ARGS
    print(args)

    # Load environment.
    env, env_grid_spec = env_suite.get_env(args['env_name'])

    # Create dataset for (offline) training.
    if args['dataset_type'] == 'dirichlet':

        sampling_dist_size = env.num_states * env.num_actions
        alpha = args['dirichlet_dataset_args']['dirichlet_alpha_coef']
        sampling_dist = np.random.dirichlet([alpha]*sampling_dist_size)

        dataset, dataset_info = _dataset_from_sampling_dist(env, env_grid_spec,
                                        sampling_dist,
                                        dataset_size=args['dataset_size'],
                                        force_full_coverage=args['force_full_coverage'])
    else:
        raise ValueError('Unkown dataset type.')

    # Store dataset.
    dataset_path = args['exp_path'] + "/dataset.json"
    print(f'Storing dataset to {dataset_path}.')
    f = open(dataset_path, "w")
    dumped = json.dumps(dataset, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    # Store dataset info.
    f = open(args['exp_path'] + "/dataset_info.json", "w")
    dumped = json.dumps(dataset_info, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()



    return dataset_path, dataset_info
