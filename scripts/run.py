import os
import json
import shutil
import pathlib


# Import scripts to create sampling distributions.
from scripts.sampling_dist_from_dirichlet import main as dirichlet_sampling_dist

# Import script to generate dataset from sampling dist.
from scripts.dataset import main as create_dataset

# Import script to train offline RL algorithm.
from scripts.train import train

# Import plots script.
from analysis.plots import main as plots

from utils.strings import create_exp_name


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

###########################################################################
VAL_ITER_DATA = {
    # 'mdp1': 'mdp1_val_iter_2021-08-27-17-49-23',
    'gridEnv1': 'gridEnv1_val_iter_2022-01-17-15-07-56',
    'gridEnv2': 'gridEnv2_val_iter_2022-01-19-11-44-01',
    # 'multiPathsEnv': 'multiPathsEnv_val_iter_2021-06-04-19-31-25',
    # 'pendulum': 'pendulum_val_iter_2021-05-24-11-48-50',
    # 'mountaincar': 'mountaincar_val_iter_v2_2021-10-20-16-17-47',
}

RUN_ARGS = {
    'env_name': 'gridEnv1',

    'dataset_args': {
        'dataset_type': 'boltzmann',

        # Number of dataset transitions.
        'dataset_size': 50_000,

        # Whether to force coverage over all (s,a) pairs, i.e.,
        # the sampling distribution always verifies p(s,a) > 0.
        'force_full_coverage': False,

        'dirichlet_dataset_args': {
            'dirichlet_alpha_coef': 100.0,
        },

        'eps_greedy_dataset_args': {
            'epsilon': 0.0,
        },

        'boltzmann_dataset_args': {
            'temperature': 0.0,
        },
    },

    'train_args':  {
        'num_runs': 4,
        'num_processors': 2,
        'num_threads_per_proc': 4, # if None then uses default settings.
        'algo': 'offline_dqn',
        'gamma': 0.9, # discount factor.
        'num_epochs': 200, # number of epochs to train.

        # Period at which checkpoints are saved.
        # (in number of epochs)
        'save_interval': 10,
        'num_rollouts': 5, # number of rollouts to execute per checkpoint.

        # Offline DQN algorithm arguments.
        'offline_dqn_args': {
            'batch_size': 100,
            'target_update_interval': 1_000,
            'learning_rate': 1e-03,
            'hidden_layers': [20,40,20],
        },

        'offline_cql_args': {
            'batch_size': 100,
            'target_update_interval': 1_000,
            'learning_rate': 1e-03,
            'hidden_layers': [20,40,20],
        }

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
    run_args['dataset_args']['val_iter_path'] = VAL_ITER_DATA[run_args['env_name']]
    dataset_path, dataset_info = create_dataset(run_args['dataset_args'])
    print('dataset info:', dataset_info)

    # Train.
    run_args['train_args']['env_name'] = run_args['env_name']
    run_args['train_args']['dataset_path'] = dataset_path
    run_args['train_args']['exp_path'] = exp_path
    train(run_args['train_args'])

    # Compute plots.
    plots(exp_name, VAL_ITER_DATA[run_args['env_name']])

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
