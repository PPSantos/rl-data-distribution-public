import os
import glob
import json
import pathlib

import gym
import numpy as np

from d3rlpy.algos.dqn import DQN
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

from envs import env_suite
from utils.json_utils import NumpyEncoder
from utils.strings import create_exp_name


DEFAULT_ARGS = {
    'env_name': 'mountaincar',

    'num_rollouts': 5,

    'n_steps': 1_000_000,
    'steps_per_epoch': 25_000,

    'start_epsilon': 0.5,
    'end_epsilon': 0.01,
    'duration_epsilon': 1_000_000,

    # DQN algorithm arguments.
    'dqn_args': {
        'gamma': 0.99, # discount factor.
        #'learning_rate': 1e-03,
        'batch_size': 100,
        #'target_update_interval': 1_000,
        'hidden_layers': [32,64,32],
    },

    # Replay buffer arguments.
    'replay_buffer_args': {
        'maxlen': 1_000_000,
    }

}

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


def train(args):

    # Setup experiment data folder.
    args['algo'] = 'dqn'
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('scripts/dqn.py arguments:')
    print(args)

    # Store args copy. 
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Load environment.
    env, _ = env_suite.get_env(args['env_name'])
    eval_env, _ = env_suite.get_env(args['env_name'])

    # Compute value iteration.
    encoder_factory = VectorEncoderFactory(
                hidden_units=args['dqn_args']['hidden_layers'],
                activation='relu')
    dqn = DQN(**args['dqn_args'], use_gpu=False,
            encoder_factory=encoder_factory)

    buffer = ReplayBuffer(**args['replay_buffer_args'], env=env)

    explorer = LinearDecayEpsilonGreedy(start_epsilon=args['start_epsilon'],
                                        end_epsilon=args['end_epsilon'],
                                        duration=args['duration_epsilon'])

    # start training
    dqn.fit_online(env,
                buffer,
                explorer=explorer,
                eval_env=eval_env,
                n_steps=args['n_steps'],
                experiment_name="0",
                n_steps_per_epoch=args['steps_per_epoch'],
                with_timestamp=False,
                logdir=exp_path,
                save_metrics=True,
                tensorboard_dir=f"{exp_path}/tb-logs/",
                update_interval=5)

    # Use checkpoints to calculate custom evaluation metrics.
    chkpt_files = glob.glob(f"{exp_path}/0/*.pt")

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
        dqn.load_model(chkpt_f)
        
        # Store rollouts mean reward.
        data['rollouts_rewards'].append(evaluate_scorer(dqn))

        # Store Q-values.
        estimated_Q_vals = np.zeros((env.num_states, env.num_actions))
        for state in range(env.num_states):
            obs = env.get_observation(state)
            for a in range(env.num_actions):
                estimated_Q_vals[state,a] = \
                    dqn.predict_value([obs], [a])[0]

        data['Q_vals'][i,:,:] = estimated_Q_vals

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps([data], cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)
    return exp_name

if __name__ == "__main__":
    exp_id = train(args=DEFAULT_ARGS)

    from analysis.plots import main as plots
    from scripts.run import VAL_ITER_DATA
    plots(exp_id=exp_id, val_iter_exp=VAL_ITER_DATA[DEFAULT_ARGS['env_name']])
