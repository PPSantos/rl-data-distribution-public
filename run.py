from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots_single import main as plots
from analysis.plot_state_qval import main as Qvalplots

# WARNING: CHOOSE THE `ARGS` DICTIONARY VARIABLE ACCORDINGLY. (e.g. oracle_fqi_args or dqn_args)

# Whether to compute plots for each experiment.
COMPUTE_PLOTS = True

# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
VAL_ITER_DATA = 'gridEnv4_val_iter_2021-04-28-09-54-18'


if __name__ == "__main__":

    exp_ids = []

    """
        One-hot observations.
    """
    """ args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = False
    args['env_args']['one_hot_obs'] = True

    max_replay_sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    # max_replay_sizes = [500_000, 300_000, 200_000, 100_000, 50_000] # gridEnv5

    for size in max_replay_sizes:
        print(f'One-hot observation, max_replay_size={size}')
        args['fqi_args']['max_replay_size'] = size
        exp_id = train(args)
        exp_ids.append(exp_id)
        if COMPUTE_PLOTS:
            plots(exp_id, VAL_ITER_DATA)
            Qvalplots(exp_id, VAL_ITER_DATA)

    print('Exp. ids:', exp_ids) """

    """
        Smoothed observations.
    """
    """ args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = True
    args['env_args']['one_hot_obs'] = False

    max_replay_sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    # max_replay_sizes = [500_000, 300_000, 200_000, 100_000, 50_000] # gridEnv5

    for size in max_replay_sizes:
        print(f'Smoothed observation, max_replay_size={size}')
        args['fqi_args']['max_replay_size'] = size
        exp_id = train(args)
        exp_ids.append(exp_id)
        if COMPUTE_PLOTS:
            plots(exp_id, VAL_ITER_DATA)
            Qvalplots(exp_id, VAL_ITER_DATA)

    print('Exp. ids:', exp_ids) """

    """
        Random observations.
    """
    args = DEFAULT_TRAIN_ARGS
    args['env_args']['smooth_obs'] = False
    args['env_args']['one_hot_obs'] = False

    max_replay_sizes = [1_000_000, 750_000, 500_000, 250_000, 100_000]
    # max_replay_sizes = [500_000, 300_000, 200_000, 100_000, 50_000] # gridEnv5

    for size in max_replay_sizes:
        print(f'Random observation, max_replay_size={size}')
        args['fqi_args']['max_replay_size'] = size
        exp_id = train(args)
        exp_ids.append(exp_id)
        if COMPUTE_PLOTS:
            plots(exp_id, VAL_ITER_DATA)
            Qvalplots(exp_id, VAL_ITER_DATA)

    print('Exp. ids:', exp_ids)
