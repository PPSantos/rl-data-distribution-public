import os
import json
import shutil
import pathlib

# Import script to generate dataset from sampling dist.
from scripts.dataset import main as create_dataset

# Import script to train offline RL algorithm.
from scripts.train import train

# Import plots script.
from analysis.plots import main as plots

from utils.strings import create_exp_name

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

###########################################################################
ORACLE_Q_VALS_DATA = {
    'gridEnv1': 'gridEnv1_q_learning_2022-02-02-19-07-14',
    'gridEnv2': 'gridEnv2_val_iter_2022-01-23-19-56-06',
    'multiPathEnv': 'multiPathEnv_val_iter_2022-01-23-18-46-25',
    'mountaincar': 'mountaincar_q_learning_2022-02-02-19-31-29',
}

RUN_ARGS = {

    # Environment name.
    'env_name': 'mountaincar',

    # scripts/dataset.py arguments.
    'dataset_args': {

        # Dataset type ('dirichlet', 'eps-greedy', or 'boltzmann').
        'dataset_type': 'dirichlet',

        # Number of dataset transitions.
        'dataset_size': 200_000,

        # Whether to force coverage over all (s,a) pairs, i.e.,
        # the sampling distribution always verifies p(s,a) > 0.
        'force_full_coverage': True,

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
            'temperature': 0.0, # [-10, 10]
        },
    },

    # scripts/train.py arguments.
    'train_args': {

        'num_runs': 4,
        'num_processors': 4,
        'algo': 'offline_dqn',
        'num_steps': 200_000, # number of learning steps (num. batch updates).
        'gamma': 0.99, # discount factor.
        'checkpoint_interval': 5_000, # period at which checkpoints are saved.
        'num_rollouts': 5, # number of rollouts to execute per checkpoint.

        # Offline DQN algorithm arguments.
        'offline_dqn_args': {
            'batch_size': 100,
            'target_update_period': 1_000,
            'learning_rate': 1e-03,
            'max_gradient_norm': None,
            'hidden_layers': [64,128,64],
        },

        # Offline CQL algorithm arguments.
        # 'offline_cql_args': {
        #     'batch_size': 100,
        #     'target_update_period': 1_000,
        #     'learning_rate': 1e-03,
        #     'max_gradient_norm': None,
        #     'hidden_layers': [32,64,32],
        # }

    },
}
###########################################################################

def main(run_args):

    # Setup experiment data folder.
    exp_name = create_exp_name({'env_name': run_args['env_name'],
                                'algo': run_args['train_args']['algo']})
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('run.py arguments:')
    print(run_args)

    # Store args copy. 
    f = open(exp_path + "/args.json", "w")
    json.dump(run_args, f)
    f.close()

    # Generate dataset.
    run_args['dataset_args']['env_name'] = run_args['env_name']
    run_args['dataset_args']['exp_path'] = exp_path
    run_args['dataset_args']['oracle_q_vals_path'] = ORACLE_Q_VALS_DATA[run_args['env_name']]
    dataset_dir, _ = create_dataset(run_args['dataset_args'])

    # Train.
    run_args['train_args']['env_name'] = run_args['env_name']
    run_args['train_args']['dataset_dir'] = dataset_dir
    run_args['train_args']['exp_path'] = exp_path
    train(run_args['train_args'])

    # Compute plots.
    plots(exp_name, ORACLE_Q_VALS_DATA[run_args['env_name']])

    # Compress and cleanup.
    shutil.make_archive(exp_path,
                    'gztar',
                    os.path.dirname(exp_path),
                    exp_name)
    shutil.rmtree(exp_path)

    print('Experiment ID:', exp_name)

    return exp_name

if __name__ == "__main__":
    main(run_args=RUN_ARGS)
