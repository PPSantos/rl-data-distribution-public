import os
import shutil

from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots import main as plots


# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
COMPUTE_PLOTS = True
VAL_ITER_DATA = 'pendulum_val_iter_2021-05-24-11-48-50'


if __name__ == "__main__":

    exp_ids = []

    """
        One-hot observations.
    """
    """ args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = False
    args['env_args']['one_hot_obs'] = True
    dict_key = args['algo'] + '_args'

    max_replay_sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    # max_replay_sizes = [500_000, 300_000, 200_000, 100_000, 50_000] # gridEnv5

    for size in max_replay_sizes:
        print(f'One-hot observation, max_replay_size={size}')

        args[dict_key]['max_replay_size'] = size
        exp_path, exp_id = train(args)
        exp_ids.append(exp_id)

        # Compute plots.
        if COMPUTE_PLOTS:
            plots(exp_id, VAL_ITER_DATA)

        # Compress and cleanup.
        shutil.make_archive(exp_path,
                        'gztar',
                        os.path.dirname(exp_path),
                        exp_id)
        shutil.rmtree(exp_path)

    print('Exp. ids:', exp_ids) """

    """
        Smoothed observations.
    """
    args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = True
    args['env_args']['one_hot_obs'] = False
    dict_key = args['algo'] + '_args'

    max_replay_sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    # max_replay_sizes = [500_000, 300_000, 200_000, 100_000, 50_000] # gridEnv5

    for size in max_replay_sizes:
        print(f'Smoothed observation, max_replay_size={size}')

        args[dict_key]['max_replay_size'] = size
        exp_path, exp_id = train(args)
        exp_ids.append(exp_id)

        # Compute plots.
        if COMPUTE_PLOTS:
            plots(exp_id, VAL_ITER_DATA)

        # Compress and cleanup.
        shutil.make_archive(exp_path,
                        'gztar',
                        os.path.dirname(exp_path),
                        exp_id)
        shutil.rmtree(exp_path)

    print('Exp. ids:', exp_ids)

    """
        Random observations.
    """
    args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = False
    args['env_args']['one_hot_obs'] = False
    dict_key = args['algo'] + '_args'

    max_replay_sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    # max_replay_sizes = [500_000, 300_000, 200_000, 100_000, 50_000] # gridEnv5

    for size in max_replay_sizes:
        print(f'Random observation, max_replay_size={size}')

        args[dict_key]['max_replay_size'] = size
        exp_path, exp_id = train(args)
        exp_ids.append(exp_id)

        # Compute plots.
        if COMPUTE_PLOTS:
            plots(exp_id, VAL_ITER_DATA)

        # Compress and cleanup.
        shutil.make_archive(exp_path,
                        'gztar',
                        os.path.dirname(exp_path),
                        exp_id)
        shutil.rmtree(exp_path)

    print('Exp. ids:', exp_ids)
