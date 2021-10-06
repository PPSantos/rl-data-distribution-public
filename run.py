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

    hidden_layers = {'gridEnv1': [20,40,20],
                     'gridEnv4': [20,40,20],
                     'multiPathsEnv': [20,40,20],
                     'pendulum': [32,64,32],
                     'mountaincar': [64,128,64]}

    for env in ['gridEnv1', 'gridEnv4', 'multiPathsEnv',
                'pendulum', 'mountaincar']:

        print('env=', env)

        # Setup train args.
        args['env_name'] = env
        args[algo_dict_key]['hidden_layers'] = hidden_layers[env]

        for a in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:

            print('alpha=', a)
            # Run.
            args[algo_dict_key]['dataset_sampling_dist_alpha'] = a
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

    exit()

    """ # Run.
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


    args['algo'] = 'dqn_e_tab'
    # Run.
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


    exit()"""

    """ deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for d in deltas:

        print('d=', d)
        # Run.
        args[algo_dict_key]['delta'] = d
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




    exit()



    layers = [[25,50,25],[32,64,32],[64,128,64]]
    for l in layers:

        print('l=', l)
        # Run.
        args[algo_dict_key]['hidden_layers'] = l
        args[algo_dict_key]['e_net_hidden_layers'] = l
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

    exit()"""

    #args[algo_dict_key]['hidden_layers'] = [20,40,20]
    #args[algo_dict_key]['e_net_hidden_layers'] = [20,40,20]

    """sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    for s in sizes:

        print('s=', s)
        # Run.
        args[algo_dict_key]['max_replay_size'] = s
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

        print('Exp. ids:', exp_ids)"""


    # Vary alpha exp.
    # (only for dqn algorithm)
    """ alphas = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    for a in alphas:

        print('a=', a)
        # Run.
        args[algo_dict_key]['synthetic_replay_buffer_alpha'] = a
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
