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
from algos.q_learning import QLearning
from algos.dqn.dqn import DQN
from algos.dqn2be.dqn2be import DQN2BE
from algos.dqn_e_tab.dqn_e_tab import DQN_E_tab
from algos.oracle_fqi.oracle_fqi import OracleFQI
from algos.fqi.fqi import FQI
from algos.linear_approximator import LinearApproximator


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/data/'

VAL_ITER_DATA = {
    'mdp1': 'mdp1_val_iter_2021-06-10-15-55-52',
    'gridEnv1': 'gridEnv1_val_iter_2021-05-14-15-54-10',
    'gridEnv4': 'gridEnv4_val_iter_2021-06-16-10-08-44',
    'multiPathsEnv': 'multiPathsEnv_val_iter_2021-06-04-19-31-25',
    'pendulum': 'pendulum_val_iter_2021-05-24-11-48-50',
    'mountaincar': 'mountaincar_val_iter_2021-06-17-13-52-11',
}

DEFAULT_TRAIN_ARGS = {
    # WARNING: 'mdp1' only works with the 'linear_approximator'
    #           (and vice versa) and 'val_iter' algorithms.

    # General arguments.
    'num_runs': 1,
    'num_processors': 1,
    'algo': 'dqn_e_tab',
    'num_episodes': 20_000,
    'gamma': 0.9, # discount factor.

    # Period at which the Q-values are stored.
    'q_vals_period': 100,

    # Period at which replay buffer counts are calculated and stored.
    'replay_buffer_counts_period': 100,

    # Whether to estimate and store the E-vals at each
    # 'q_vals_period' steps for the current set of Q-values.
    'compute_e_vals': True,

    # Env. arguments.
    'env_args': {
        'env_name': 'gridEnv1',
        'dim_obs': 8, # (for grid env. only).
        'time_limit': 50, # (for grid env. only).
        'tabular': False, # (for grid env. only).
        'smooth_obs': False, # (for grid env. only).
        'one_hot_obs': False, # (for grid env. only).
    },

    # Evaluation rollouts arguments.
    'rollouts_period': 100,
    'num_rollouts': 5,
    'rollouts_types': ['default', 'stochastic_actions_1', 'stochastic_actions_2',
                    'stochastic_actions_3', 'stochastic_actions_4', 'stochastic_actions_5',
                    'uniform_init_state_dist'],

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

    # Linear function approximator arguments.
    'linear_approximator_args': {
        'alpha': 1e-04,
        'expl_eps_init': 0.9,
        'expl_eps_final': 0.0,
        'expl_eps_episodes': 3_500,
        'synthetic_replay_buffer': True,
        'synthetic_replay_buffer_alpha': 1_000,
        'replay_size': 20_000,
        'batch_size': 100,
    },

    # DQN arguments.
    # (Standard DQN algorithm).
    'dqn_args': {
        'batch_size': 100,
        'target_update_period': 1_000,
        'samples_per_insert': 25.0,
        'min_replay_size': 50_000,
        'max_replay_size': 1_000_000,
        'epsilon_init': 1.0,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 1_000_000,
        'learning_rate': 1e-03,
        'hidden_layers': [20,40,20],
        'synthetic_replay_buffer': False,
        'synthetic_replay_buffer_alpha': 1_000,
    },

    # DQN + E_tab arguments.
    # (DQN version featuring tabular E-values driven exploration).
    'dqn_e_tab_args': {
        # Default DQN args.
        'batch_size': 100,
        'target_update_period': 1_000,
        'samples_per_insert': 25.0,
        'min_replay_size': 50_000,
        'max_replay_size': 1_000_000,
        'learning_rate': 1e-03,
        'hidden_layers': [20,40,20],
        # E-learning args.
        'lr_lambda': 0.05,
        'epsilon_init': 1.0,
        'epsilon_final': 0.0,
        'epsilon_schedule_episodes': 20_000,
    },

    # DQN-2BE arguments.
    # TODO: this will be dqn_e_func
    'dqn2be_args': {
        # Default DQN args.
        'batch_size': 100,
        'target_update_period': 1_000,
        'samples_per_insert': 25.0,
        'min_replay_size': 50_000,
        'max_replay_size': 1_000_000,
        'prioritized_replay': False,
        'importance_sampling_exponent': 0.9,
        'priority_exponent': 0.6,
        'epsilon_init': 1.0,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 1_000_000,
        'learning_rate': 1e-03,
        'hidden_layers': [20,40,20],
        # Bellman error learning args.
        'e_net_hidden_layers': [20,40,20],
        'e_net_learning_rate': 1e-03,
        'target_e_net_update_period': 1_000,
        # 'delta_init': 0.25,
        # 'delta_final': 0.25,
        # 'delta_schedule_timesteps': 10_000_000,
    },

    # FQI arguments.
    'fqi_args': {
        'batch_size': 100,
        'num_sampling_steps': 1_000,
        'num_gradient_steps': 20,
        'max_replay_size': 1_000_000,
        'epsilon_init': 0.9,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 500_000,
        'learning_rate': 1e-03,
        'hidden_layers': [20,40,20],
        'reweighting_type': 'default', # default, actions or full.
        'synthetic_replay_buffer': True,
        'synthetic_replay_buffer_alpha': 1_000,
    },

    # Oracle FQI arguments.
    'oracle_fqi_args': {
        'batch_size': 100,
        'num_sampling_steps': 1_000,
        'num_gradient_steps': 20,
        'max_replay_size': 1_000_000,
        'epsilon_init': 0.9,
        'epsilon_final': 0.0,
        'epsilon_schedule_timesteps': 500_000,
        'learning_rate': 1e-03,
        'hidden_layers': [20,40,20],
        'reweighting_type': 'default', # default, actions or full.
        'synthetic_replay_buffer': True,
        'synthetic_replay_buffer_alpha': 1_000,
    }

}

def create_exp_name(args):
    return args['env_args']['env_name'] + \
        '_' + args['algo'] + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def train_run(run_args):

    time_delay, exp_path, args = run_args
    log_path = exp_path + f'/logs_learner_{time_delay}'

    time.sleep(time_delay)

    # Load train environment.
    if args['env_args']['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys():
        env, env_grid_spec = env_suite.get_custom_grid_env(**args['env_args'], env_type='default',
                                                        absorb=False, seed=time_delay)
    else:
        env = env_suite.get_env(args['env_args']['env_name'])
        env_grid_spec = None

    # Load rollouts environment(s).
    if args['env_args']['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys():
        rollouts_envs = [env_suite.get_custom_grid_env(**args['env_args'], env_type=t,
                                                    absorb=False, seed=time_delay)[0]
                            for t in args['rollouts_types']]
    else:
        rollouts_envs = [env_suite.get_env(args['env_args']['env_name'])]

    # print('Env num states:', env.num_states)
    # print('Env num actions:', env.num_actions)
    # print('Env transition matrix shape:', env.transition_matrix().shape)
    # print('Env initial state distribution:', env.initial_state_distribution)
    # print('Env render:')
    # env.reset()
    # env.render()
    # print('\n')

    # Load optimal (oracle) policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + VAL_ITER_DATA[args['env_args']['env_name']]
    print(f"Opening experiment {VAL_ITER_DATA[args['env_args']['env_name']]} to get oracle Q-vals")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        val_iter_data = json.load(f)
        val_iter_data = json.loads(val_iter_data)
        val_iter_data = val_iter_data[0]
    f.close()

    # Instantiate algorithm.
    if args['algo'] == 'val_iter':
        args['val_iter_args']['gamma'] = args['gamma']
        agent = ValueIteration(env, **args['val_iter_args'])

    elif args['algo'] == 'q_learning':

        raise ValueError('Not implemented')

        args['q_learning_args']['gamma'] = args['gamma']
        agent = QLearning(env, **args['q_learning_args'])

    elif args['algo'] == 'linear_approximator':
        args['linear_approximator_args']['gamma'] = args['gamma']
        agent = LinearApproximator(env, env_grid_spec, **args['linear_approximator_args'])

    elif args['algo'] == 'dqn':
        args['dqn_args']['oracle_q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]
        args['dqn_args']['discount'] = args['gamma']
        agent = DQN(env, env_grid_spec, log_path, args['dqn_args'])

    elif args['algo'] == 'dqn2be':

        raise ValueError('Not implemented - this will be dqn_e_func algorithm')

        args['dqn2be_args']['oracle_q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]
        args['dqn2be_args']['discount'] = args['gamma']
        agent = DQN2BE(env, env_grid_spec, log_path, args['dqn2be_args'])

    elif args['algo'] == 'dqn_e_tab':
        args['dqn_e_tab_args']['oracle_q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]
        args['dqn_e_tab_args']['discount'] = args['gamma']
        agent = DQN_E_tab(env, env_grid_spec, log_path, args['dqn_e_tab_args'])

    elif args['algo'] == 'fqi':
        args['fqi_args']['discount'] = args['gamma']
        agent = FQI(env, env_grid_spec, args['fqi_args'])

    elif args['algo'] == 'oracle_fqi':
        args['oracle_fqi_args']['oracle_q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]
        args['oracle_fqi_args']['discount'] = args['gamma']
        agent = OracleFQI(env, env_grid_spec, args['oracle_fqi_args'])

    else:
        raise ValueError("Unknown algorithm.")

    # Train agent.
    train_data = agent.train(num_episodes=args['num_episodes'],
                             q_vals_period=args['q_vals_period'],
                             replay_buffer_counts_period=args['replay_buffer_counts_period'],
                             num_rollouts=args['num_rollouts'],
                             rollouts_period=args['rollouts_period'],
                             rollouts_envs=rollouts_envs,
                             compute_e_vals=args['compute_e_vals'])

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

    if DEFAULT_TRAIN_ARGS['algo'] != 'val_iter':

        from analysis.plots import main as plots
        env_name = DEFAULT_TRAIN_ARGS['env_args']['env_name']
        val_iter_data = VAL_ITER_DATA[env_name]

        # Compute plots.
        plots(exp_id, val_iter_data)

        # Compress and cleanup.
        shutil.make_archive(exp_path,
                        'gztar',
                        os.path.dirname(exp_path),
                        exp_id)
        shutil.rmtree(exp_path)
