import os
import json
import pathlib

from envs import env_suite
from utils.json_utils import NumpyEncoder
from utils.strings import create_exp_name
from algos.q_learning import QLearning

DEFAULT_ARGS = {
    'env_name': 'gridEnv1',
    'gamma': 0.9, # discount factor.
    'alpha': 0.1, # learning rate.

    # Eps-greedy exploration parameters.
    'expl_eps_init': 1.0,
    'expl_eps_final': 0.0,
    'expl_eps_episodes': 18_000,
    'num_episodes': 20_000,

    # Replay buffer parameters.
    'replay_buffer_size': 50_000,
    'replay_buffer_batch_size': 128,

    'learning_steps': 20_000, # train_offline number of learning steps
}

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


def main(args=None):

    # Setup train args.
    if args is None:
        args = DEFAULT_ARGS
    args['algo'] = 'q_learning'

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('scripts/q_learning.py arguments:')
    print(args)

    # Store args copy. 
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Load environment.
    env, _ = env_suite.get_env(args['env_name'])

    # Compute value iteration.
    algo = QLearning(env=env, qlearning_args=args)
    train_data = algo.train_offline(args['learning_steps'])

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)

if __name__ == "__main__":
    main()
