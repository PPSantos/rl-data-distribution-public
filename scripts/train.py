import os
import glob
import json
import time
import numpy as np
import pathlib
import multiprocessing as mp
import multiprocessing.context as ctx
ctx._force_start_method('spawn')

from tqdm import tqdm
import tensorflow as tf
from acme import specs
from acme.utils import loggers

from envs import env_suite, grid_spec
from utils.json_utils import NumpyEncoder
from utils import tf2_layers
from envs.utils import wrap_env, run_rollout

# Import algorithms.
from algos.dqn.offline_agent import OfflineDQN
from algos.cql.offline_agent import OfflineCQL


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


def train_run(run_args):

    time_delay, args = run_args

    time.sleep(time_delay)

    # Set random seeds.
    tf.random.set_seed(time_delay)
    np.random.seed(time_delay)

    # Load environment.
    env, env_grid_spec = env_suite.get_env(args['env_name'])
    env = wrap_env(env)
    env_spec = specs.make_environment_spec(env)

    # Read and prepare dataset.
    batch_size = args[args['algo']+'_args']['batch_size']
    dataset = tf.data.experimental.load(args['dataset_dir'])
    dataset = dataset.repeat().shuffle(len(dataset)).batch(batch_size,
                                                drop_remainder=True)
    dataset = dataset.prefetch(2)

    # Load algorithm (learner).
    if args['algo'] == 'offline_dqn':
        network = tf2_layers.create_MLP(env_spec,
            args['offline_dqn_args']['hidden_layers'])
        offline_agent = OfflineDQN(
            env_spec=env_spec,
            network=network,
            dataset=dataset,
            target_update_period=args['offline_dqn_args']['target_update_period'],
            learning_rate=args['offline_dqn_args']['learning_rate'],
            discount=args['gamma'],
            logger=loggers.TerminalLogger(label='agent_logger',
                        time_delta=5., print_fn=print),
            max_gradient_norm=args['offline_dqn_args']['max_gradient_norm'],
            checkpoint=True,
            checkpoint_interval=args['checkpoint_interval'],
            save_directory=args['exp_path'] + f'/{time_delay}',
        )

    elif args['algo'] == 'offline_cql':

        network = tf2_layers.create_MLP(env_spec,
            args['offline_cql_args']['hidden_layers'])
        offline_agent = OfflineCQL(
            env_spec=env_spec,
            network=network,
            dataset=dataset,
            target_update_period=args['offline_cql_args']['target_update_period'],
            learning_rate=args['offline_cql_args']['learning_rate'],
            discount=args['gamma'],
            alpha=args['offline_cql_args']['alpha'],
            logger=loggers.TerminalLogger(label='agent_logger',
                        time_delta=5., print_fn=print),
            max_gradient_norm=args['offline_cql_args']['max_gradient_norm'],
            checkpoint=True,
            checkpoint_interval=args['checkpoint_interval'],
            save_directory=args['exp_path'] + f'/{time_delay}',
        )
    else:
        raise ValueError('Unknown algorithm.')

    # Train.
    print('Started training.')
    for step in tqdm(range(args['num_steps'])):
        offline_agent.step()
        if step % args['checkpoint_interval'] == 0:
            # Execute evaluation rollout.
            print('Rollout reward:', run_rollout(env, offline_agent))
    print('Finished training.')

    # Use checkpoints to calculate evaluation metrics.
    chkpt_files = glob.glob(f"{args['exp_path']}/{time_delay}/*.index")

    steps = [os.path.split(p)[1].split('.')[0] for p in chkpt_files]
    steps = [int(p.split('_')[1]) for p in steps]
    chkpt_files = [os.path.splitext(x)[0] for _, x in sorted(zip(steps, chkpt_files))]
    steps = sorted(steps)
    print('chkpt_files:', chkpt_files)

    data = {}
    data['steps'] = steps
    data['Q_vals'] = np.zeros((len(steps),
                    env.num_states, env.num_actions))
    data['rollouts_rewards'] = []

    for i, chkpt_f in enumerate(chkpt_files):

        # Load checkpoint.
        offline_agent.load(chkpt_f)
        
        # Store rollouts mean reward.
        data['rollouts_rewards'].append(np.mean([run_rollout(env, offline_agent)
                                for _ in range(args['num_rollouts'])]))

        # Store Q-values.
        for state in range(env.num_states):
            if env_grid_spec:
                xy = env_grid_spec.idx_to_xy(state)
                tile_type = env_grid_spec.get_value(xy, xy=True)
                if tile_type == grid_spec.WALL:
                    data['Q_vals'][i,state,:] = 0.0
                else:
                    obs = env.get_observation(state)
                    data['Q_vals'][i,state,:] = offline_agent.get_Q_vals(
                        np.array(obs, dtype=np.float32))
            else:
                obs = env.get_observation(state)
                data['Q_vals'][i,state,:] = offline_agent.get_Q_vals(
                    np.array(obs, dtype=np.float32))

    return data


def train(args):

    print('\nRunning scripts/train.py.')

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
