import os
import json
import pathlib

from envs import env_suite
from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder
from algos.value_iteration import ValueIteration

DEFAULT_ARGS = {
    'env_name': 'gridEnv1',
    'gamma': 0.9, # discount factor.
    'epsilon': 0.001, # error threshold.
}

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


def main(args=None):

    # Setup train args.
    if args is None:
        args = DEFAULT_ARGS
    args['algo'] = 'val_iter'

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('scripts/value_iteration.py arguments:')
    print(args)

    # Store args copy. 
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Load environment.
    env, _ = env_suite.get_env(args['env_name'])

    # Compute value iteration.
    algo = ValueIteration(env=env, gamma=args['gamma'], epsilon=args['epsilon'])
    train_data = algo.compute()

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)

if __name__ == "__main__":
    main()
