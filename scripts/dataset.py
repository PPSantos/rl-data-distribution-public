import os
import json
import pathlib
import numpy as np
import scipy.stats
import collections

import tensorflow as tf

from utils.json_utils import NumpyEncoder
from envs import env_suite, grid_spec
from utils.array_functions import build_eps_greedy_policy, build_boltzmann_policy


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

DEFAULT_DATASET_ARGS = {

    # Environment name.
    'env_name': 'gridEnv1',

    # Dataset type ('dirichlet', 'eps-greedy', or 'boltzmann').
    'dataset_type': 'boltzmann',

    # Number of dataset transitions.
    'dataset_size': 50_000,

    # Whether to force coverage over all (s,a) pairs, i.e.,
    # the sampling distribution always verifies p(s,a) > 0.
    'force_full_coverage': False,

    # dataset_type=dirichlet args.
    'dirichlet_dataset_args': {
        'dirichlet_alpha_coef': 100.0, # [0.0, 100.0]
    },

    # dataset_type=eps-greedy args.
    'eps_greedy_dataset_args': {
        'epsilon': 0.0, # [0.0, 1.0]
    },

    # dataset_type=boltzmann args.
    'boltzmann_dataset_args': {
        'temperature': 0.0, # [-10.0, 10.0]
    },
}

def _dict_to_namedtuple(typename, data):
    return collections.namedtuple(typename, data.keys())(
        *(_dict_to_namedtuple(k, v) if isinstance(v, dict) else v
            for k, v in data.items())
    )

def _add_transition(dataset, transition):

    # Unpack transition.
    obs, action, reward, next_obs, done, info = transition

    # The discount is 0.0 if done is True and the episode was not truncated.
    # Otherwise the discount is 1.0 (default value).
    discount = float(not(done and (not info.get('TimeLimit.truncated', False))))
    dataset['data']['observation'].append(np.array(obs, dtype=np.float32))
    dataset['data']['action'].append(np.array(action, dtype=np.int32))
    dataset['data']['reward'].append(np.array(reward, dtype=np.float32))
    dataset['data']['discount'].append(np.array(discount, dtype=np.float32))
    dataset['data']['next_observation'].append(np.array(next_obs, dtype=np.float32))

def _calculate_dataset_dist_from_counts(env, env_grid_spec, sa_counts):

    if env_grid_spec:
        # Flag walls with nan values.
        for state in range(env.num_states):
            for action in range(env.num_actions):
                xy = env_grid_spec.idx_to_xy(state)
                tile_type = env_grid_spec.get_value(xy, xy=True)
                if tile_type == grid_spec.WALL:
                    sa_counts[state, action] = np.nan

    sa_counts = sa_counts.flatten() # [S*A]
    sa_counts = sa_counts[np.logical_not(np.isnan(sa_counts))] # remove nans
    dataset_dist = sa_counts / np.sum(sa_counts) # [S*A]

    return dataset_dist, sa_counts

def _add_missing_transitions(env, env_grid_spec, dataset, sa_counts):

    env.reset()

    zero_positions = np.where(sa_counts == 0)
    print('Number of missing (s,a) pairs:', np.sum((sa_counts == 0)))

    for (state, action) in zip(*zero_positions):

        # Skip walls.
        if env_grid_spec:
            xy = env_grid_spec.idx_to_xy(state)
            tile_type = env_grid_spec.get_value(xy, xy=True)
            if tile_type == grid_spec.WALL:
                continue

        observation = env.get_observation(state)

        # Sample next state, observation and reward.
        env.set_state(state)
        next_observation, reward, done, info = env.step(action)

        transition = (observation, action, reward, next_observation, done, info)
        _add_transition(dataset, transition)

        sa_counts[state, action] += 1

        env.reset()

def _dataset_from_sampling_dist(env, env_grid_spec, sampling_dist: np.ndarray,
                dataset_size: int, force_full_coverage: bool):

    dataset = {'data': { 'observation': [], 'action': [], 'reward': [],
                'discount': [], 'next_observation': [],}, 'info': {}}

    mesh = np.array(np.meshgrid(np.arange(env.num_states),
                                np.arange(env.num_actions)))
    sa_combinations = mesh.T.reshape(-1, 2)
    sa_counts = np.zeros((env.num_states, env.num_actions))

    env.reset()

    for _ in range(dataset_size):

        # Randomly sample (state, action) pair according to sampling dist.
        if env_grid_spec:
            tile_type = grid_spec.WALL
            while tile_type == grid_spec.WALL:
                sampled_idx = np.random.choice(np.arange(len(sampling_dist)), p=sampling_dist)
                state, action = sa_combinations[sampled_idx]
                xy = env_grid_spec.idx_to_xy(state)
                tile_type = env_grid_spec.get_value(xy, xy=True)
        else:
            sampled_idx = np.random.choice(np.arange(len(sampling_dist)), p=sampling_dist)
            state, action = sa_combinations[sampled_idx]

        sa_counts[state, action] += 1
        observation = env.get_observation(state)

        # Sample next state, observation and reward.
        env.set_state(state)
        next_observation, reward, done, info = env.step(action)

        transition = (observation, action, reward, next_observation, done, info)
        _add_transition(dataset, transition)

        env.reset()

    if force_full_coverage:
        # Correct dataset such that we have coverage over all (state, action) pairs.
        _add_missing_transitions(env, env_grid_spec,
                dataset=dataset, sa_counts=sa_counts)

    dataset_dist, sa_counts = _calculate_dataset_dist_from_counts(
                            env, env_grid_spec, sa_counts)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(_dict_to_namedtuple('Dataset', dataset))

    # Store dataset info.
    dataset_info = {}
    dataset_info['dataset_dist'] = dataset_dist
    dataset_info['dataset_sa_counts'] = sa_counts
    dataset_info['dataset_entropy'] = scipy.stats.entropy(dataset_dist)

    return dataset, dataset_info

def _dataset_from_policy(env, env_grid_spec, policy,
                            dataset_size: int, force_full_coverage: bool):

    dataset = {'data': { 'observation': [], 'action': [], 'reward': [],
                'discount': [], 'next_observation': [],}, 'info': {}}

    # Rollout policy.
    episode_rewards = []
    num_samples = 0

    # Matrix to store (s,a) counts.
    sa_counts = np.zeros((env.num_states, env.num_actions))

    while num_samples < dataset_size:

        observation = env.reset()
        state = env.get_state()

        done = False
        episode_cumulative_reward = 0
        while not done:

            # Pick action.
            action = policy(state)

            # Env step.
            next_observation, reward, done, info = env.step(action)
            next_state = env.get_state()

            # Log data.
            episode_cumulative_reward += reward
            sa_counts[state, action] += 1

            transition = (observation, action, reward, next_observation, done, info)
            _add_transition(dataset, transition)

            state = next_state
            observation = next_observation
            num_samples += 1

        episode_rewards.append(episode_cumulative_reward)
    
    if force_full_coverage:
        # Correct dataset such that we have coverage over all (state, action) pairs.
        _add_missing_transitions(env, env_grid_spec,
                dataset=dataset, sa_counts=sa_counts)

    dataset_dist, sa_counts = _calculate_dataset_dist_from_counts(
                        env, env_grid_spec, sa_counts)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(_dict_to_namedtuple('Dataset', dataset))

    print('Average policy reward:', np.mean(episode_rewards))

    dataset_info = {}
    dataset_info['episode_rewards'] = episode_rewards
    dataset_info['dataset_sa_counts'] = sa_counts
    dataset_info['dataset_dist'] = dataset_dist
    dataset_info['dataset_entropy'] = scipy.stats.entropy(dataset_dist)

    return dataset, dataset_info


def main(args=None):

    print('\nRunning scripts/dataset.py.')

    if not args:
        args = DEFAULT_DATASET_ARGS
    print(args)

    # Load environment.
    env, env_grid_spec = env_suite.get_env(args['env_name'])

    # Create dataset for (offline) training.
    print('Creating dataset.')
    if args['dataset_type'] == 'dirichlet':

        sampling_dist_size = env.num_states * env.num_actions
        alpha = args['dirichlet_dataset_args']['dirichlet_alpha_coef']
        sampling_dist = np.random.dirichlet([alpha]*sampling_dist_size)

        dataset, dataset_info = _dataset_from_sampling_dist(env, env_grid_spec,
                                    sampling_dist=sampling_dist,
                                    dataset_size=args['dataset_size'],
                                    force_full_coverage=args['force_full_coverage'])

    elif args['dataset_type'] in ('eps-greedy', 'boltzmann'):

        oracle_q_vals_path = DATA_FOLDER_PATH + args['oracle_q_vals_path']
        print(f"Opening experiment {args['oracle_q_vals_path']}")
        with open(oracle_q_vals_path + "/train_data.json", 'r') as f:
            oracle_q_vals_data = json.load(f)
            oracle_q_vals_data = json.loads(oracle_q_vals_data)
            oracle_q_vals_data = oracle_q_vals_data
        f.close()
        optimal_q_vals = np.array(oracle_q_vals_data['Q_vals']) # [S,A]

        if args['dataset_type'] == 'eps-greedy':
            policy = build_eps_greedy_policy(optimal_q_vals,
                    epsilon=args['eps_greedy_dataset_args']['epsilon'])
        else:
            policy = build_boltzmann_policy(optimal_q_vals,
                    temperature=args['boltzmann_dataset_args']['temperature'])

        dataset, dataset_info = _dataset_from_policy(env, env_grid_spec,
                                    policy=policy,
                                    dataset_size=args['dataset_size'],
                                    force_full_coverage=args['force_full_coverage'])

    else:
        raise ValueError('Unknown dataset type.')

    # Store tf.dataset.
    print('Dataset info:\n', dataset_info)
    dataset_dir = args['exp_path'] + "/dataset/"
    os.makedirs(dataset_dir, exist_ok=True)
    print(f'Storing dataset to {dataset_dir}')
    tf.data.experimental.save(dataset, dataset_dir)

    # Store dataset info.
    f = open(args['exp_path'] + "/dataset_info.json", "w")
    dumped = json.dumps(dataset_info, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return dataset_dir, dataset_info
