import os
import shutil
import numpy as np

from train import train
from train import DEFAULT_TRAIN_ARGS, VAL_ITER_DATA
from analysis.plots import main as plots

from utils.sampling_dist import DEFAULT_SAMPLING_DIST_ARGS, _build_policy_func
from utils.sampling_dist import main as sampling_dist

if __name__ == "__main__":

    exp_ids = []
    sampling_dist_ids = []

    # Load args.
    args = DEFAULT_TRAIN_ARGS
    algo_dict_key = args['algo'] + '_args'

    hidden_layers = {'gridEnv1': [20,40,20],
                     'gridEnv4': [20,40,20],
                     'multiPathsEnv': [20,40,20],
                     'pendulum': [32,64,32],
                     'mountaincar': [64,128,64]}

    sampling_dist_args = DEFAULT_SAMPLING_DIST_ARGS

    for env in ['gridEnv4',]: #['gridEnv1', 'gridEnv4', 'multiPathsEnv',
               # 'pendulum', 'mountaincar']:

        print('env=', env)
        args['env_args']['env_name'] = env
        sampling_dist_args['env_args']['env_name'] = env

        val_iter_data = VAL_ITER_DATA[env]
        print('val_iter_data', val_iter_data)

        args[algo_dict_key]['hidden_layers'] = hidden_layers[env]

        NUM_SAMPLING_DISTS = 50
        for _ in range(NUM_SAMPLING_DISTS):

            # Create sampling dist.
            policy = _build_policy_func(num_switched_actions=np.random.randint(0,64),
                                    epsilon=np.random.uniform(0,1))

            sampling_dist_path, sampling_dist_id = sampling_dist(args=sampling_dist_args, policy=policy)
            sampling_dist_ids.append(sampling_dist_id)

            # Run.
            args[algo_dict_key]['dataset_custom_sampling_dist'] = sampling_dist_path
            exp_path, exp_id = train(args)
            exp_ids.append(exp_id)
            # Compute plots.
            plots(exp_id, val_iter_data)
            # Compress and cleanup.
            shutil.make_archive(exp_path,
                                'gztar',
                                os.path.dirname(exp_path),
                                exp_id)
            shutil.rmtree(exp_path)

            print('Exp. ids:', exp_ids)
            print('Sampling dist. ids:', sampling_dist_ids)
