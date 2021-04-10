import os
import json
import numpy as np
import pathlib

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'

EXP_IDS = ['8_8_q_learning_2021-04-09-18-04-25']

VAL_ITER_DATA = '8_8_val_iter_2021-04-09-18-08-36'


def calculate_CI_bootstrap(x_hat, samples, num_resamples=20000):
    """
        Calculates 95 % interval using bootstrap.
        REF: https://ocw.mit.edu/courses/mathematics/
            18-05-introduction-to-probability-and-statistics-spring-2014/
            readings/MIT18_05S14_Reading24.pdf
    """
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    means = np.mean(resampled, axis=0)
    diffs = means - x_hat
    bounds = [x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95)]

    return bounds


if __name__ == "__main__":

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Load data.
    data = {}
    for exp_name in EXP_IDS:

        exp_path = DATA_FOLDER_PATH + exp_name
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            exp_data = json.loads(exp_data)
        f.close()

        # Average data for each train run.
        averaged_data = {}

        # episode_rewards
        r_concatenated = np.array([exp['episode_rewards'] for exp in exp_data])
        r_concatenated = np.mean(r_concatenated, axis=0)
        averaged_data['episode_rewards'] = r_concatenated

        # epsilon_values
        e_concatenated = np.array([exp['epsilon_values'] for exp in exp_data])
        e_concatenated = np.mean(e_concatenated, axis=0)
        averaged_data['epsilon_values'] = e_concatenated

        # Q_vals
        q_concatenated = np.array([exp['Q_vals'] for exp in exp_data])
        q_concatenated = np.mean(q_concatenated, axis=0)
        averaged_data['Q_vals'] = q_concatenated

        data[exp_name] = averaged_data

    if VAL_ITER_DATA:
        # Load optimal Q-values.
        val_iter_path = DATA_FOLDER_PATH + VAL_ITER_DATA
        with open(val_iter_path + "/train_data.json", 'r') as f:
            val_iter_data = json.load(f)
            val_iter_data = json.loads(val_iter_data)
            val_iter_data = val_iter_data[0]
        f.close()

    """
        Episode rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp_id in EXP_IDS:
        Y = data[exp_id]['episode_rewards']
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X,Y,label=exp_id)

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.savefig('{0}/episode_rewards.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Epsilon.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp_id in EXP_IDS:
        Y = data[exp_id]['epsilon_values']
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X,Y,label=exp_id)

    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.savefig('{0}/episode_epsilon.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_epsilon.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()


    """
        Q-values plots.
    """
    if VAL_ITER_DATA:

        # Prepare data to plot.
        val_iter_Q_vals = np.array(val_iter_data['Q_vals'])
        exp_Q_vals = {}
        for exp_id in EXP_IDS:
            exp_Q_vals[exp_id] = np.array(data[exp_id]['Q_vals'])

        # Sum plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for exp_id in EXP_IDS:

            Y = np.sum(np.abs(val_iter_Q_vals - exp_Q_vals[exp_id]), axis=(1,2))
            X = np.linspace(1, len(Y), len(Y))

            plt.plot(X,Y,label=exp_id)

        plt.xlabel('Episode')
        plt.ylabel('Q-values error')
        plt.title('np.sum(np.abs(val_iter_Q_vals - exp_Q_vals))')

        plt.legend()

        plt.savefig('{0}/q_values_summed_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/q_values_summed_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Mean + Std/CI plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for exp_id in EXP_IDS:

            Q_abs_diffs = np.abs(val_iter_Q_vals - exp_Q_vals[exp_id])
            Y = np.mean(Q_abs_diffs, axis=(1,2))
            X = np.linspace(1, len(Y), len(Y))
            # Y_std = np.std(Q_abs_diffs, axis=(1,2))

            # CI calculation.
            Q_abs_diffs_flatten = Q_abs_diffs.reshape(len(Y), -1)
            X_CI = np.arange(0, len(Y), 100)
            CI_bootstrap = [calculate_CI_bootstrap(Y[x], Q_abs_diffs_flatten[x,:]) for x in X_CI]
            CI_bootstrap = np.array(CI_bootstrap).T
            CI_bootstrap = np.flip(CI_bootstrap, axis=0)
            CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

            p = plt.plot(X,Y,label=exp_id)
            plt.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)
            # plt.fill_between(X, Y-Y_std, Y+Y_std, color=p[0].get_color(), alpha=0.15)

        plt.xlabel('Episode')
        plt.ylabel('Q-values error')
        plt.title('np.mean(np.abs(val_iter_Q_vals - exp_Q_vals))')

        plt.legend()

        plt.savefig('{0}/q_values_mean_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/q_values_mean_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Distribution plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        val_iter_Q_vals_flattened = val_iter_Q_vals.flatten()

        for exp_id in EXP_IDS:

            exp_Q_vals_flattened = exp_Q_vals[exp_id].reshape(exp_Q_vals[exp_id].shape[0], -1)

            abs_diff = np.abs(val_iter_Q_vals_flattened - exp_Q_vals_flattened)
            X = np.arange(0, len(abs_diff), 500)
            abs_diff_list = abs_diff[X,:].tolist() # Sub-sample.
            abs_diff_mean = np.mean(abs_diff[X,:], axis=1) # Sub-sample.

            violin = plt.violinplot(abs_diff_list, positions=X,
                                    showextrema=True, widths=350)

            plt.scatter(X, abs_diff_mean, label=exp_id, s=12)

        plt.xlabel('Episode')
        plt.ylabel('Q-values error')
        plt.title('Distribution of np.abs(val_iter_Q_vals - exp_Q_vals)')

        plt.legend()

        plt.savefig('{0}/q_values_violinplot_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/q_values_violinplot_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.close()

    """ def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError() """

    """ #print('\nQ-vals:', Q_vals)
        print('Max Q-vals:', max_Q_vals)
        print('Policy:', policy)
        env.render()

        # Plot policy.
        print('Policy:')
        size_x = args['env_args']['size_x']
        size_y = args['env_args']['size_y']
        sys.stdout.write('-'*(size_x+2)+'\n')
        for h in range(size_y):
            sys.stdout.write('|')
            for w in range(size_x):
                sys.stdout.write(str(policy[xy_to_idx((w,h), size_x, size_y)]))
            sys.stdout.write('|\n')
        sys.stdout.write('-' * (size_x + 2)+'\n')

        # Plot max Q-values.
        print('Max Q-vals:')
        for h in range(size_y):
            for w in range(size_x):
                sys.stdout.write("{:.1f} ".format(max_Q_vals[xy_to_idx((w,h),size_x, size_y)]))
            sys.stdout.write('\n')
        sys.stdout.write('\n')
    """
