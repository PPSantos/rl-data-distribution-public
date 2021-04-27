import sys
import os
import json
import time
import numpy as np
import pathlib
from datetime import datetime
import multiprocessing as mp

from rlutil.json_utils import NumpyEncoder
from envs import env_suite

from algos.value_iteration import ValueIteration
from algos.q_learning import QLearning
from algos.dqn.dqn import DQN
from algos.oracle_fqi.oracle_fqi import OracleFQI

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/data/'

DEFAULT_TRAIN_ARGS = {

    # General arguments.
    'num_runs': 5,
    'num_processors': 5,
    'algo': 'oracle_fqi',
    'num_episodes': 10_000,
    'gamma': 0.9,

    # Env. arguments.
    'env_args': {
        'env_name': 'gridEnv1',
        'dim_obs': 8,
        'time_limit': 50,
        'tabular': False,
        'smooth_obs': False,
        'one_hot_obs': True,
    },

    # Value iteration arguments.
    'val_iter_args': {
        'epsilon': 0.05
    },

    # Q-learning arguments.
    'q_learning_args': {
        'alpha': 0.05,
        'expl_eps_init': 0.9,
        'expl_eps_final': 0.01,
        'expl_eps_episodes': 9_000,
    },

    # DQN arguments.
    'dqn_args': {
        'batch_size': 128,
        'prefetch_size': 2,
        'target_update_period': 5_000,
        'samples_per_insert': 128.0,
        'min_replay_size': 50_000,
        'max_replay_size': 500_000,
        'prioritized_replay': False,
        'importance_sampling_exponent': 0.9,
        'priority_exponent': 0.6,
        'n_step': 1,
        'epsilon_init': 0.9,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 450_000,
        'learning_rate': 1e-03,
        'hidden_layers': [10,10],
    },

    # Oracle FQI arguments.
    'oracle_fqi_args': {
        'oracle_q_vals': '8_8_val_iter_2021-04-13-17-55-56', # exp. id of val-iter/oracle Q-vals.
        'batch_size': 100,
        'prefetch_size': 1,
        'num_sampling_steps': 1_000,
        'num_gradient_steps': 20,
        'max_replay_size': 500_000,
        'n_step': 1,
        'epsilon_init': 0.9,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 200_000,
        'learning_rate': 1e-03,
        'hidden_layers': [10,10],
    }

}

def create_exp_name(args):
    return args['env_args']['env_name'] + \
        '_' + args['algo'] + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def train_run(run_args):

    time_delay, args = run_args[0], run_args[1]

    time.sleep(time_delay)

    # Load environment.
    env, env_grid_spec = env_suite.get_custom_env(**args['env_args'], absorb=False)

    # print('Env num states:', env.num_states)
    # print('Env num actions:', env.num_actions)
    # print('Env transition matrix shape:', env.transition_matrix().shape)
    # print('Env initial state distribution:', env.initial_state_distribution)
    print('Env render:')
    env.reset()
    env.render()
    print('\n')

    # Instantiate algorithm.
    if args['algo'] == 'val_iter':
        args['val_iter_args']['gamma'] = args['gamma']
        agent = ValueIteration(env, **args['val_iter_args'])

    elif args['algo'] == 'q_learning':
        args['q_learning_args']['gamma'] = args['gamma']
        agent = QLearning(env, **args['q_learning_args'])

    elif args['algo'] == 'dqn':
        args['dqn_args']['discount'] = args['gamma']
        agent = DQN(env, env_grid_spec, args['dqn_args'])

    elif args['algo'] == 'oracle_fqi':

        # Load optimal (oracle) policy/Q-values.
        val_iter_path = DATA_FOLDER_PATH + args['oracle_fqi_args']['oracle_q_vals']
        print(f"Opening experiment {args['oracle_fqi_args']['oracle_q_vals']} to get oracle Q-vals")
        with open(val_iter_path + "/train_data.json", 'r') as f:
            val_iter_data = json.load(f)
            val_iter_data = json.loads(val_iter_data)
            val_iter_data = val_iter_data[0]
        f.close()

        args['oracle_fqi_args']['oracle_q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]
        args['oracle_fqi_args']['discount'] = args['gamma']
        agent = OracleFQI(env, env_grid_spec, args['oracle_fqi_args'])

    else:
        raise ValueError("Unknown algorithm.")

    # Train agent.
    train_data = agent.train(num_episodes=args['num_episodes'])

    return train_data


def train(train_args=None):

    # Setup train args.
    if train_args is None:
        args = DEFAULT_TRAIN_ARGS
    else:
        args = train_args

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    print('\nExperiment ID:', exp_name)
    print('train.py arguments:')
    print(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Adjust the number of processors if necessary.
    if args['num_processors'] > mp.cpu_count():
        args['num_processors'] = mp.cpu_count()
        print(f"Downgraded the number of processors to {args['num_processors']}.")

    # Train agent(s).
    with mp.Pool(processes=args['num_processors']) as pool:
        train_data = pool.map(train_run, [(2*t, args) for t in range(args['num_runs'])])
        pool.close()
        pool.join()

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return exp_name

if __name__ == "__main__":
    train() # Uses DEFAULT_TRAIN_ARGS.
