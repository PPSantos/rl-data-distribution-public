
import os
import json
import glob
import pathlib
import time
import numpy as np
import multiprocessing as mp

from acme import specs
from acme.utils import loggers

from algos.dqn.agent import DQN
from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder
from utils.tf2_layers import create_MLP
from envs.utils import wrap_env, run_rollout
from utils.online_learning_loop import EnvironmentLoop
from envs import env_suite

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


DEFAULT_TRAIN_ARGS = {

    'num_processors': 4,
    'num_runs': 4,

    'env_name': 'mountaincar',

    'num_steps': 200_000,
    'num_rollouts': 5,

    'dqn_args': {
        'batch_size': 100,
        'prefetch_size': 1,
        'target_update_period': 1_000,
        'samples_per_insert': 100.0,
        'min_replay_size': 10_000,
        'max_replay_size': 200_000,
        'n_step': 1,
        'epsilon_init': 0.2,
        'epsilon_final': 0.01,
        'epsilon_schedule_timesteps': 200_000,
        'learning_rate': 1e-03,
        'discount': 0.9,
        'max_gradient_norm': None,
        'checkpoint': True,
        'checkpoint_interval': 5_000,
        'hidden_layers': [32,64,32],
    }
}


def train_run(run_args):

    time_delay, args = run_args
    time.sleep(time_delay)

    env, _ = env_suite.get_env(args['env_name'])
    env = wrap_env(env)
    env_spec = specs.make_environment_spec(env)

    dqn_args = args['dqn_args']
    network = create_MLP(env_spec, dqn_args['hidden_layers'])
    agent = DQN(env_spec, network,
        batch_size=dqn_args['batch_size'],
        prefetch_size=dqn_args['prefetch_size'],
        target_update_period=dqn_args['target_update_period'],
        samples_per_insert=dqn_args['samples_per_insert'],
        min_replay_size=dqn_args['min_replay_size'],
        max_replay_size=dqn_args['max_replay_size'],
        n_step=dqn_args['n_step'],
        epsilon_init=dqn_args['epsilon_init'],
        epsilon_final=dqn_args['epsilon_final'],
        epsilon_schedule_timesteps=dqn_args['epsilon_schedule_timesteps'],
        learning_rate=dqn_args['learning_rate'],
        discount=dqn_args['discount'],
        logger=loggers.TerminalLogger(label='agent_logger',
                        time_delta=10., print_fn=print),
        max_gradient_norm=None,
        checkpoint=dqn_args['checkpoint'],
        checkpoint_interval=dqn_args['checkpoint_interval'],
        save_directory=args['exp_path'] + f'/{time_delay}'
    )

    env_loop = EnvironmentLoop(env, agent,
            logger=loggers.TerminalLogger(label='env_logger', 
                            time_delta=2., print_fn=print))
    print('Started training.')
    env_loop.run(num_steps=args['num_steps'])
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
        agent.load(chkpt_f)

        # Store rollouts mean reward.
        data['rollouts_rewards'].append(np.mean([run_rollout(env, agent)
                                for _ in range(args['num_rollouts'])]))

        # Store Q-values.
        for state in range(env.num_states):
            obs = env.get_observation(state)
            data['Q_vals'][i,state,:] = agent.get_Q_vals(np.array(obs, dtype=np.float32))

    return data

def train(args):

    # Setup experiment data folder.
    args['algo'] = 'dqn'
    exp_id = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_id
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_id)
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
    args['exp_path'] = exp_path
    with mp.Pool(processes=args['num_processors']) as pool:
        train_data = pool.map(train_run, [(2*t, args) for t in range(args['num_runs'])])
        pool.close()
        pool.join()

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Exp id:', exp_id)

    return exp_id


if __name__ == '__main__':

    exp_id = train(args=DEFAULT_TRAIN_ARGS)

    from analysis.plots import main as plots
    from scripts.run import ORACLE_Q_VALS_DATA
    plots(exp_id=exp_id, val_iter_exp=ORACLE_Q_VALS_DATA[DEFAULT_TRAIN_ARGS['env_name']])
