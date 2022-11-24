import os
import json
import pathlib

from utils.json_utils import NumpyEncoder
from four_state_mdp.env import FourStateMDP

from utils.strings import create_exp_name
from algos.value_iteration import ValueIteration

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

ARGS = {
    "env_name": "four_state_mdp",
    "algo": "val_iter", 

    "gamma": 1.0, # Discount factor.
    "epsilon": 0.001, # Threshold for termination.
}

if __name__ == "__main__":

    # Setup experiment data folder.
    exp_name = create_exp_name({'env_name': ARGS['env_name'],
                                'algo': ARGS['algo']})
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('run_value_iteration.py arguments:')
    print(ARGS)

    env = FourStateMDP()

    val_iter = ValueIteration(env, gamma=ARGS["gamma"], epsilon=ARGS["epsilon"])

    data = val_iter.compute()
    print(data)

    # Store data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)
