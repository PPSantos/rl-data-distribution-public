"""
    Helper script that generates a sampling distribution by sampling from a Dirichlet distribution.
"""
import os
import json
import numpy as np
import pathlib
import scipy.stats

from envs import env_suite
from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

DEFAULT_ARGS = {
    'env_name': 'gridEnv1',
    'dirichlet_alpha_coef': 10.0,
}

def main(args=None):

    if not args:
        args = DEFAULT_ARGS
    args['algo'] = 'sampling_dist'

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('scripts/sampling_dist_from_dirichlet.py arguments:')
    print(args)

    # Load environment.
    env, _ = env_suite.get_env(args['env_name'], seed=0)

    # Create sampling dist.
    sampling_dist_size = env.num_states * env.num_actions
    sampling_dist = np.random.dirichlet([args['dirichlet_alpha_coef']]*sampling_dist_size)
    print('(S,A) dist. entropy:', scipy.stats.entropy(sampling_dist))

    data = {}
    data['sampling_dist'] = sampling_dist

    # Store data.
    f = open(exp_path + "/data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return exp_path + "/data.json", exp_name


if __name__ == "__main__":
    main()
