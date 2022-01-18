"""
    Helper script that generates a sampling distribution by sampling from a Dirichlet distribution.
"""
import random
import numpy as np
import pathlib
import scipy.stats

from envs import env_suite

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

DEFAULT_ARGS = {
    'env_name': 'gridEnv1',
    'dirichlet_alpha_coef': 10.0,
}

def main(args=None):

    if not args:
        args = DEFAULT_ARGS
    args['algo'] = 'sampling_dist'

    print(args)

    # Load environment.
    env, _ = env_suite.get_env(args['env_name'])

    # Create sampling dist.
    sampling_dist_size = env.num_states * env.num_actions
    sampling_dist = np.random.dirichlet([args['dirichlet_alpha_coef']]*sampling_dist_size)

    sampling_dist_info = {}
    sampling_dist_info['sampling_dist'] = sampling_dist
    sampling_dist_info['sampling_dist_entropy'] = scipy.stats.entropy(sampling_dist)

    return sampling_dist, sampling_dist_info


if __name__ == "__main__":
    main()
