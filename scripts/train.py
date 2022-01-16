import os
import glob
import json
import time
import shutil
import numpy as np
import pathlib
import multiprocessing as mp
from typing import List

from envs import env_suite, grid_spec
from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder

# Import algorithms.
from d3rlpy.algos import DQN
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.dataset import Transition
from d3rlpy.models.encoders import VectorEncoderFactory


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

VAL_ITER_DATA = {
    # 'mdp1': 'mdp1_val_iter_2021-08-27-17-49-23',
    'gridEnv1': 'gridEnv1_val_iter_2022-01-15-21-58-32',
    # 'gridEnv4': 'gridEnv4_val_iter_2021-06-16-10-08-44',
    # 'multiPathsEnv': 'multiPathsEnv_val_iter_2021-06-04-19-31-25',
    # 'pendulum': 'pendulum_val_iter_2021-05-24-11-48-50',
    # 'mountaincar': 'mountaincar_val_iter_v2_2021-10-20-16-17-47',
}

DEFAULT_TRAIN_ARGS = {

    # General arguments.
    'num_runs': 1,
    'num_processors': 1,
    'env_name': 'gridEnv1',
    'algo': 'offline_dqn',
    'gamma': 0.9, # discount factor.
    'num_epochs': 200, # number of epochs to train.

    # Period at which checkpoints are saved.
    # (in number of epochs)
    'save_interval': 10,
    'num_rollouts': 5, # number of rollouts to execute per checkpoint.

    'dataset_args': {
        # Number of dataset transitions.
        'dataset_size': 50_000,

        # Absolute path to json data file containing
        # sampling dist.
        'custom_sampling_dist': None,

        # Alpha coefficient of the Dirichlet distribution
        # used to generate/sample sampling distributions.
        # Ignored if `dataset_custom_sampling_dist` is set.
        'sampling_dist_alpha': 1000.0,

        # Whether to force coverage over all (s,a) pairs, i.e.,
        # the sampling distribution always verifies p(s,a) > 0.
        'force_full_coverage': True,
    },

    # Offline DQN algorithm arguments.
    'offline_dqn_args': {
        'batch_size': 100,
        'target_update_interval': 1_000,
        'learning_rate': 1e-03,
        'hidden_layers': [32,64,32],
    },
}


def create_dataset(env, env_grid_spec, sampling_dist: np.ndarray,
                dataset_size: int, force_full_coverage: bool) -> List[Transition]:

    print('Creating dataset.')

    transitions : List[Transition] = []

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

        sa_counts[state,action] += 1
        observation = env.get_observation(state)

        # Sample next state, observation and reward.
        env.set_state(state)
        next_observation, reward, done, info = env.step(action)

        transition = Transition(observation_shape=observation.shape,
                                action_size=env.num_actions,
                                observation=observation,
                                action=action,
                                reward=reward,
                                next_observation=next_observation,
                                terminal=0.0)

        transitions.append(transition)

        env.reset()

    if force_full_coverage:
        # Correct dataset such that we have coverage over all (state, action) pairs.
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

            transition = Transition(observation_shape=observation.shape,
                                    action_size=env.num_actions,
                                    observation=observation,
                                    action=action,
                                    reward=reward,
                                    next_observation=next_observation,
                                    terminal=0.0)

            transitions.append(transition)

            env.reset()

    return transitions

def train_run(run_args):

    time_delay, exp_path, args = run_args

    time.sleep(time_delay)

    # Load environment.
    env, env_grid_spec = env_suite.get_env(args['env_name'], seed=time_delay)
    
    # Create dataset for (offline) training.
    dataset_args = args['dataset_args']

    if dataset_args['custom_sampling_dist'] is None:
        sampling_dist_size = env.num_states * env.num_actions
        sampling_dist = np.random.dirichlet(
                        [dataset_args['sampling_dist_alpha']]*sampling_dist_size)
    else:
        raise ValueError('Custom sampling distribution not implement.')

    dataset = create_dataset(env, env_grid_spec, sampling_dist,
                dataset_size=dataset_args['dataset_size'],
                force_full_coverage=dataset_args['force_full_coverage'])

    # Load algorithm.
    if args['algo'] == 'offline_dqn':
        print('h_layers:', args['offline_dqn_args']['hidden_layers'])
        encoder_factory = VectorEncoderFactory(
                    hidden_units=args['offline_dqn_args']['hidden_layers'],
                    activation='relu')
        args['offline_dqn_args']['gamma'] = args['gamma']
        algo = DQN(**args['offline_dqn_args'],
                encoder_factory=encoder_factory)
    else:
        raise ValueError('Unknown algorithm.')

    # Train.
    algo.build_with_env(env)
    algo.fit(dataset,
            n_epochs=args['num_epochs'],
            save_metrics=True,
            save_interval=args['save_interval'],
            experiment_name=f"{time_delay}",
            with_timestamp=False,
            logdir=exp_path,
            tensorboard_dir=f"{exp_path}/tb-logs/")

    # Use checkpoints to calculate custom evaluation metrics.
    chkpt_files = glob.glob(f"{exp_path}/{time_delay}/*.pt")

    steps = [os.path.split(p)[1].split('.')[0] for p in chkpt_files]
    steps = [int(p.split('_')[1]) for p in steps]
    chkpt_files = [x for _, x in sorted(zip(steps, chkpt_files))]
    steps = sorted(steps)

    evaluate_scorer = evaluate_on_environment(env,
                    n_trials=args['num_rollouts'], epsilon=0.0)

    data = {}
    data['steps'] = steps
    data['Q_vals'] = np.zeros((len(steps),
                    env.num_states, env.num_actions))
    data['rollouts_rewards'] = []

    for i, chkpt_f in enumerate(chkpt_files):

        # Load checkpoint.
        algo.load_model(chkpt_f)
        
        # Store rollouts mean reward.
        data['rollouts_rewards'].append(evaluate_scorer(algo))

        # Store Q-values.
        estimated_Q_vals = np.zeros((env.num_states, env.num_actions))
        for state in range(env.num_states):
            xy = env_grid_spec.idx_to_xy(state)
            tile_type = env_grid_spec.get_value(xy, xy=True)
            if tile_type == grid_spec.WALL:
                estimated_Q_vals[state,:] = 0.0
            else:
                obs = env.get_observation(state)
                for a in range(env.num_actions):
                    estimated_Q_vals[state,a] = \
                        algo.predict_value([obs], [a])[0]

        data['Q_vals'][i,:,:] = estimated_Q_vals

    return data


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
