import sys
import os
import json
import time
import shutil
import numpy as np
import pathlib
from datetime import datetime
import multiprocessing as mp

from rlutil.json_utils import NumpyEncoder
from envs import env_suite

# Import algorithms.
from algos.value_iteration import ValueIteration
from algos.value_iteration_v2 import ValueIterationV2
from algos.dqn.dqn import DQN
from algos.dqn.offline_dqn import OfflineDQN
from algos.q_learning import QLearning

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/data/'

VAL_ITER_DATA = {
    'mdp1': 'mdp1_val_iter_2021-08-27-17-49-23',
    'gridEnv1': 'gridEnv1_val_iter_2021-05-14-15-54-10',
    'gridEnv4': 'gridEnv4_val_iter_2021-06-16-10-08-44',
    'multiPathsEnv': 'multiPathsEnv_val_iter_2021-06-04-19-31-25',
    'pendulum': 'pendulum_val_iter_2021-05-24-11-48-50',
    'mountaincar': 'mountaincar_val_iter_v2_2021-10-20-16-17-47',
}

DEFAULT_TRAIN_ARGS = {

    # General arguments.
    'num_runs': 5,
    'num_processors': 5,
    'algo': 'q_learning',
    'gamma': 0.95, # discount factor.

    # Environment arguments.
    'env_name': 'mountaincar',

    # Period at which the Q-values are stored.
    # (the period can either be number of steps
    # or episodes, depending on the algorihm)
    'q_vals_period': 1_000,

    # Period at which replay buffer counts are stored.
    # (the period can either be number of steps
    # or episodes, depending on the algorihm)
    'replay_buffer_counts_period': 1_000,

    # Evaluation rollouts arguments.
    # (the period can either be number of steps
    # or episodes, depending on the algorihm)
    'rollouts_period': 1_000,
    'num_rollouts': 5,

    # Value iteration arguments.
    'val_iter_args': {
        'epsilon': 0.5
    },

    # Standard DQN algorithm arguments.
    'dqn_args': {
        'num_learning_episodes': 1_000,
        'batch_size': 100,
        'target_update_period': 1_000,
        'samples_per_insert': 25.0,
        'min_replay_size': 20_000,
        'max_replay_size': 1_000_000,
        'epsilon_init': 1.0,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 1_000_000,
        'learning_rate': 1e-03,
        'hidden_layers': [20,40,20],
    },

    # Offline DQN algorithm arguments.
    'offline_dqn_args': {
        'batch_size': 100,
        'target_update_period': 1_000,
        'learning_rate': 1e-03,
        'num_learning_steps': 200_000,
        'hidden_layers': [32,64,32],
        'dataset_size': 50_000,

        # Full/absolute path to json data file containing
        # sampling dist.
        'dataset_custom_sampling_dist': None,

        # Alpha coefficient of the Dirichlet distribution
        # used to generate/sample sampling distributions.
        # Ignored if `dataset_custom_sampling_dist` is set.
        'dataset_sampling_dist_alpha': 1000.0,

        # Whether to force coverage over all (s,a) pairs, i.e.,
        # the sampling distribution always verifies p(s,a) > 0.
        'dataset_force_full_coverage': True,
    },

    # Q-learning arguments.
    'q_learning_args': {
        'alpha': 0.1,
        'expl_eps_init': 1.0,
        'expl_eps_final': 0.0,
        'expl_eps_episodes': 18_000,
        'num_episodes': 20_000,

        'replay_buffer_size': 1_000_000,
        'replay_buffer_batch_size': 128,
    },

}

def create_exp_name(args):
    return args['env_name'] + \
        '_' + args['algo'] + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def train_run(run_args):

    time_delay, exp_path, args = run_args
    log_path = exp_path + f'/logs_learner_{time_delay}'

    time.sleep(time_delay)

    # Load train (and rollouts) environment.
    env, env_grid_spec, rollouts_envs = env_suite.get_env(args['env_name'], seed=time_delay)

    # Instantiate algorithm.
    if args['algo'] == 'val_iter':
        args['val_iter_args']['gamma'] = args['gamma']
        agent = ValueIteration(env, **args['val_iter_args'])
    elif args['algo'] == 'val_iter_v2':
        args['val_iter_args']['gamma'] = args['gamma']
        agent = ValueIterationV2(env, **args['val_iter_args'])
    elif args['algo'] == 'dqn':
        args['dqn_args']['discount'] = args['gamma']
        agent = DQN(env, env_grid_spec, log_path, args['dqn_args'])
    elif args['algo'] == 'offline_dqn':
        args['offline_dqn_args']['discount'] = args['gamma']
        agent = OfflineDQN(env, env_grid_spec, log_path, args['offline_dqn_args'])
    elif args['algo'] == 'q_learning':
        args['q_learning_args']['gamma'] = args['gamma']
        agent = QLearning(env, args['q_learning_args'])
    else:
        raise ValueError("Unknown algorithm.")

    # Train agent.
    train_data = agent.train(q_vals_period=args['q_vals_period'],
                             rollouts_period=args['rollouts_period'],
                             num_rollouts=args['num_rollouts'],
                             rollouts_envs=rollouts_envs,
                             replay_buffer_counts_period=args['replay_buffer_counts_period'])

    return train_data


def train(train_args=None):

    # Setup train args.
    if train_args is None:
        args = DEFAULT_TRAIN_ARGS
    else:
        args = train_args

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('train.py arguments:')
    print(args)

    # Store args copy. 
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Adjust the number of processors if necessary.
    if args['num_processors'] > mp.cpu_count():
        args['num_processors'] = mp.cpu_count()
        print(f"Downgraded the number of processors to {args['num_processors']}.")

    # Train agent(s).
    with mp.Pool(processes=args['num_processors']) as pool:
        train_data = pool.map(train_run, [(2*t, exp_path, args) for t in range(args['num_runs'])])
        pool.close()
        pool.join()

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return exp_path, exp_name

if __name__ == "__main__":

    # Train (uses DEFAULT_TRAIN_ARGS).
    exp_path, exp_id = train()

    if DEFAULT_TRAIN_ARGS['algo'] not in ('val_iter', 'val_iter_v2'):

        from analysis.plots import main as plots
        env_name = DEFAULT_TRAIN_ARGS['env_name']
        val_iter_data = VAL_ITER_DATA[env_name]

        # Compute plots.
        plots(exp_id, val_iter_data)

        # Compress and cleanup.
        shutil.make_archive(exp_path,
                        'gztar',
                        os.path.dirname(exp_path),
                        exp_id)
        shutil.rmtree(exp_path)
