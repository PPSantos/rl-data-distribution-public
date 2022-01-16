"""
    Experimental setup 1: Vary the expected entropy of the sampling distribution. 
"""
import os
import shutil

from train import train
from train import DEFAULT_TRAIN_ARGS, VAL_ITER_DATA
from analysis.plots import main as plots

ENVS = ['gridEnv1', 'gridEnv2']

HIDDEN_LAYERS = {
    'gridEnv1': [20,40,20],
    'gridEnv2': [20,40,20],
    # 'multiPathsEnv': [20,40,20],
    # 'pendulum': [32,64,32],
    # 'mountaincar': [64,128,64]
}

if __name__ == "__main__":

    print('RUNNING scripts/exp_setup_1.py')

    exp_ids = []

    # Load args.
    args = DEFAULT_TRAIN_ARGS
    algo_dict_key = args['algo'] + '_args'

    for env in ENVS:

        print('env=', env)

        # Setup train args.
        args['env_name'] = env
        args[algo_dict_key]['hidden_layers'] = HIDDEN_LAYERS[env]

        for a in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:

            print('Alpha=', a)

            # Run.
            args['dataset_args']['sampling_dist_alpha'] = a
            exp_path, exp_id = train(args)
            exp_ids.append(exp_id)

            # Compute plots.
            plots(exp_id, VAL_ITER_DATA[env])

            # Compress and cleanup.
            shutil.make_archive(exp_path,
                                'gztar',
                                os.path.dirname(exp_path),
                                exp_id)
            shutil.rmtree(exp_path)

            print('Exp. ids:', exp_ids)
