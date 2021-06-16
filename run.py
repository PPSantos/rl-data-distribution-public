import os
import shutil

from train import train
from train import DEFAULT_TRAIN_ARGS, VAL_ITER_DATA
from analysis.plots import main as plots


if __name__ == "__main__":

    exp_ids = []

    # Load args.
    args = DEFAULT_TRAIN_ARGS
    algo_dict_key = args['algo'] + '_args'
    env_name = args['env_args']['env_name']
    val_iter_data = VAL_ITER_DATA[env_name]
    print('val_iter_data', val_iter_data)

    # Setup max_replay_sizes variable.
    if env_name in ['gridEnv1', 'gridEnv2', 'gridEnv3', 'gridEnv4', 'pendulum']:
        max_replay_sizes = [1_000_000, 750_000, 500_000] #, 250_000, 100_000]
    elif env_name in ['gridEnv5']:
        max_replay_sizes = [500_000, 375_000, 250_000, 125_000, 50_000]
    elif env_name in ['mountaincar']:
        max_replay_sizes = [2_000_000]
    else:
        raise ValueError('Error.')
    print('max_replay_sizes', max_replay_sizes)

    if env_name in ['gridEnv1', 'gridEnv2', 'gridEnv3', 'gridEnv4', 'gridEnv5']:

        """
            GridEnv + One-hot observations.
        """
        """ args['env_args']['smooth_obs'] = False
        args['env_args']['one_hot_obs'] = True

        for size in max_replay_sizes:
            print(f'GridEnv + One-hot observation, max_replay_size={size}')

            # Run.
            args[algo_dict_key]['max_replay_size'] = size
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

        print('Exp. ids:', exp_ids) """

        """
            GridEnv + Smoothed observations.
        """
        args['env_args']['smooth_obs'] = True
        args['env_args']['one_hot_obs'] = False

        args[algo_dict_key]['synthetic_replay_buffer'] = False

        for size in max_replay_sizes:

            # Run.
            args[algo_dict_key]['max_replay_size'] = size
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

        args[algo_dict_key]['synthetic_replay_buffer'] = True

        for size in max_replay_sizes:

            # Run.
            args[algo_dict_key]['max_replay_size'] = size
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


        """
            GridEnv + Random observations.
        """
        """ args['env_args']['smooth_obs'] = False
        args['env_args']['one_hot_obs'] = False

        for size in max_replay_sizes:
            print(f'GridEnv + Random observation, max_replay_size={size}')
            
            # Run.
            args[algo_dict_key]['max_replay_size'] = size
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

        print('Exp. ids:', exp_ids) """


    elif env_name in ['pendulum', 'mountaincar']:

        """
            Pendulum and mountain car envs.
        """
        args[algo_dict_key]['synthetic_replay_buffer'] = False

        for size in max_replay_sizes:
            print(f'{env_name} env., max_replay_size={size}')

            # Run.
            args[algo_dict_key]['max_replay_size'] = size
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

        args[algo_dict_key]['synthetic_replay_buffer'] = True

        for size in max_replay_sizes:
            print(f'{env_name} env., max_replay_size={size}')

            # Run.
            args[algo_dict_key]['max_replay_size'] = size
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


    else:
        raise ValueError('Error.')