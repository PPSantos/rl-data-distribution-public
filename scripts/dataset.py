import json
import pathlib
import numpy as np
import scipy.stats

from utils.json_utils import NumpyEncoder
from envs import env_suite, grid_spec
from utils.array_functions import build_eps_greedy_policy, build_boltzmann_policy


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

DEFAULT_DATASET_ARGS = {

    # Environment name.
    'env_name': 'gridEnv1',

    # Dataset type (dirichlet, eps-greedy, or boltzmann).
    'dataset_type': 'boltzmann',

    # Number of dataset transitions.
    'dataset_size': 50_000,

    # Whether to force coverage over all (s,a) pairs, i.e.,
    # the sampling distribution always verifies p(s,a) > 0.
    'force_full_coverage': False,

    # type=dirichlet args.
    'dirichlet_dataset_args': {
        'dirichlet_alpha_coef': 5.0,
    },

    # type=eps-greedy args.
    'eps_greedy_dataset_args': {
        'epsilon': 0.0,
    },

    # type=boltzmann args.
    'boltzmann_dataset_args': {
        'temperature': 0.0,
    },
}

def _calculate_dataset_dist_from_counts(env, env_grid_spec, sa_counts):

    if env_grid_spec:
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

    return dataset_dist, sa_counts

def _add_missing_transitions(env, env_grid_spec, transitions, sa_counts):

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
        next_observation, reward, _, _ = env.step(action)

        transitions.append((observation, action, reward, next_observation))

        sa_counts[state,action] += 1

        env.reset()

def _dataset_from_sampling_dist(env, env_grid_spec, sampling_dist: np.ndarray,
                dataset_size: int, force_full_coverage: bool):

    transitions = []

    mesh = np.array(np.meshgrid(np.arange(env.num_states),
                                np.arange(env.num_actions)))
    sa_combinations = mesh.T.reshape(-1, 2)
    sa_counts = np.zeros((env.num_states, env.num_actions))

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

        transitions.append((observation, action, reward, next_observation))

        env.reset()

    if force_full_coverage:
        # Correct dataset such that we have coverage over all (state, action) pairs.
        _add_missing_transitions(env, env_grid_spec,
                transitions=transitions, sa_counts=sa_counts)

    dataset_dist, sa_counts = _calculate_dataset_dist_from_counts(
                            env, env_grid_spec, sa_counts)

    dataset_info = {}
    dataset_info['dataset_dist'] = dataset_dist
    dataset_info['dataset_sa_counts'] = sa_counts
    dataset_info['dataset_entropy'] = scipy.stats.entropy(dataset_dist)

    return transitions, dataset_info

def _dataset_from_policy(env, env_grid_spec, policy,
                            dataset_size: int, force_full_coverage: bool):

    transitions = []

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
            next_observation, reward, done, _ = env.step(action)
            next_state = env.get_state()

            # Log data.
            episode_cumulative_reward += reward
            sa_counts[state,action] += 1

            transitions.append((observation, action,
                                reward, next_observation))

            state = next_state
            observation = next_observation
            num_samples += 1

        episode_rewards.append(episode_cumulative_reward)
    
    if force_full_coverage:
        # Correct dataset such that we have coverage over all (state, action) pairs.
        _add_missing_transitions(env, env_grid_spec,
                transitions=transitions, sa_counts=sa_counts)

    dataset_dist, sa_counts = _calculate_dataset_dist_from_counts(
                        env, env_grid_spec, sa_counts)

    # print('Average policy reward:', np.mean(episode_rewards))

    dataset_info = {}
    dataset_info['episode_rewards'] = episode_rewards
    dataset_info['dataset_sa_counts'] = sa_counts
    dataset_info['dataset_dist'] = dataset_dist
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
    print('Creating dataset.')
    if args['dataset_type'] == 'dirichlet':

        sampling_dist_size = env.num_states * env.num_actions
        alpha = args['dirichlet_dataset_args']['dirichlet_alpha_coef']
        sampling_dist = np.random.dirichlet([alpha]*sampling_dist_size)

        dataset, dataset_info = _dataset_from_sampling_dist(env, env_grid_spec,
                                    sampling_dist=sampling_dist,
                                    dataset_size=args['dataset_size'],
                                    force_full_coverage=args['force_full_coverage'])

    elif args['dataset_type'] == 'eps-greedy':

        val_iter_path = DATA_FOLDER_PATH + args['val_iter_path']
        print(f"Opening experiment {args['val_iter_path']}")
        with open(val_iter_path + "/train_data.json", 'r') as f:
            val_iter_data = json.load(f)
            val_iter_data = json.loads(val_iter_data)
            val_iter_data = val_iter_data
        f.close()
        optimal_q_vals = np.array(val_iter_data['Q_vals']) # [S,A]

        policy = build_eps_greedy_policy(optimal_q_vals,
                epsilon=args['eps_greedy_dataset_args']['epsilon'])

        dataset, dataset_info = _dataset_from_policy(env, env_grid_spec,
                                    policy=policy,
                                    dataset_size=args['dataset_size'],
                                    force_full_coverage=args['force_full_coverage'])

    elif args['dataset_type'] == 'boltzmann':

        val_iter_path = DATA_FOLDER_PATH + args['val_iter_path']
        print(f"Opening experiment {args['val_iter_path']}")
        with open(val_iter_path + "/train_data.json", 'r') as f:
            val_iter_data = json.load(f)
            val_iter_data = json.loads(val_iter_data)
            val_iter_data = val_iter_data
        f.close()
        optimal_q_vals = np.array(val_iter_data['Q_vals']) # [S,A]

        policy = build_boltzmann_policy(optimal_q_vals,
                temperature=args['boltzmann_dataset_args']['temperature'])

        dataset, dataset_info = _dataset_from_policy(env, env_grid_spec,
                                    policy=policy,
                                    dataset_size=args['dataset_size'],
                                    force_full_coverage=args['force_full_coverage'])

    else:
        raise ValueError('Unknown dataset type.')

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
