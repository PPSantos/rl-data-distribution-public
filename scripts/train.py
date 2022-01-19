import os
import glob
import json
import time
import numpy as np
import pathlib
from typing import List
import multiprocessing as mp


from envs import env_suite, grid_spec
from utils.json_utils import NumpyEncoder

# Import algorithms.
from d3rlpy.algos import DQN, DiscreteCQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.dataset import Transition


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


def _read_dataset(env, dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        dataset = json.loads(dataset)
    f.close()

    transitions : List[Transition]= []

    for transition in dataset:
        observation = np.array(transition[0])
        action = np.int(transition[1])
        reward = np.float(transition[2])
        next_observation = np.array(transition[3])
        transition = Transition(observation_shape=observation.shape,
                                action_size=env.num_actions,
                                observation=observation,
                                action=action,
                                reward=reward,
                                next_observation=next_observation,
                                terminal=0.0)

        transitions.append(transition)

    return transitions

def train_run(run_args):

    time_delay, args = run_args

    time.sleep(time_delay)

    # Load environment.
    env, env_grid_spec = env_suite.get_env(args['env_name'])

    # Read dataset.
    dataset = _read_dataset(env, args['dataset_path'])

    # Load algorithm.
    if args['algo'] == 'offline_dqn':
        encoder_factory = VectorEncoderFactory(
                    hidden_units=args['offline_dqn_args']['hidden_layers'],
                    activation='relu')
        args['offline_dqn_args']['gamma'] = args['gamma']
        algo = DQN(**args['offline_dqn_args'], use_gpu=False,
                encoder_factory=encoder_factory)
    elif args['algo'] == 'offline_cql':
        encoder_factory = VectorEncoderFactory(
                    hidden_units=args['offline_cql_args']['hidden_layers'],
                    activation='relu')
        args['offline_cql_args']['gamma'] = args['gamma']
        algo = DiscreteCQL(**args['offline_cql_args'], use_gpu=False,
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
            logdir=args['exp_path'],
            tensorboard_dir=f"{args['exp_path']}/tb-logs/")

    # Use checkpoints to calculate custom evaluation metrics.
    chkpt_files = glob.glob(f"{args['exp_path']}/{time_delay}/*.pt")

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


def train(args):

    print('\nRunning scripts/train.py.')
    #print(args)

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
    f = open(args['exp_path'] + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return train_data
