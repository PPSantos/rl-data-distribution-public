import os
import sys
import math
import json
import random
import numpy as np
import pathlib

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import statsmodels.api as sm

FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'

# VAL_ITER_DATA = '8_8_val_iter_2021-04-13-10-09-46' # Env-8,8-1
VAL_ITER_DATA = '8_8_val_iter_2021-04-13-17-27-39' # Env-8,8-2

# VAL_ITER_DATA = '8_8_val_iter_2021-04-12-08-53-47' #8,8,walls

EXPS_DATA = [
            # {'id': '8_8_q_learning_2021-04-13-10-25-28', 'label': 'Q-learning'}, # Env-8,8-1
            {'id': '8_8_q_learning_2021-04-13-17-44-22', 'label': 'Q-learning'}, # Env-8,8-2
            
            #{'id': '8_8_q_learning_2021-04-12-00-34-49', 'label': 'Q-learning'}, #8,8,no walls
            #{'id': '8_8_q_learning_2021-04-12-00-43-35', 'label': 'Q-learning'}, #8,8, walls

            #{'id': '8_8_dqn_2021-04-11-11-16-35', 'label': 'DQN+1hot+500k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-11-34-37', 'label': 'DQN+1hot+400k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-11-52-38', 'label': 'DQN+1hot+300k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-12-10-49', 'label': 'DQN+1hot+200k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-12-28-48', 'label': 'DQN+1hot+100k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-12-47-05', 'label': 'DQN+1hot+50k'},  #8,8,no walls

            #{'id': '8_8_dqn_2021-04-11-13-53-49', 'label': 'DQN+smooth+500k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-14-12-11', 'label': 'DQN+smooth+400k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-14-30-14', 'label': 'DQN+smooth+300k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-14-48-09', 'label': 'DQN+smooth+200k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-15-06-07', 'label': 'DQN+smooth+100k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-15-24-06', 'label': 'DQN+smooth+50k'},  #8,8,no walls

            #{'id': '8_8_dqn_2021-04-11-17-29-29', 'label': 'DQN+rand+500k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-17-47-22', 'label': 'DQN+rand+400k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-18-05-20', 'label': 'DQN+rand+300k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-18-23-27', 'label': 'DQN+rand+200k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-18-41-34', 'label': 'DQN+rand+100k'}, #8,8,no walls
            #{'id': '8_8_dqn_2021-04-11-19-00-43', 'label': 'DQN+rand+50k'}, #8,8,no walls

            # {'id': '8_8_dqn_2021-04-12-01-00-07', 'label': 'DQN+1hot+500k'}, #8,8,walls
            # {'id': '8_8_dqn_2021-04-12-01-18-48', 'label': 'DQN+1hot+400k'}, #8,8,walls
            # {'id': '8_8_dqn_2021-04-12-01-37-13', 'label': 'DQN+1hot+300k'}, #8,8,walls
            # {'id': '8_8_dqn_2021-04-12-01-55-38', 'label': 'DQN+1hot+200k'}, #8,8,walls
            # {'id': '8_8_dqn_2021-04-12-02-14-20', 'label': 'DQN+1hot+100k'}, #8,8,walls
            # {'id': '8_8_dqn_2021-04-12-02-33-06', 'label': 'DQN+1hot+50k'},  #8,8,walls
            # {'id': '8_8_dqn_2021-04-12-02-51-56', 'label': 'DQN+1hot+25k'},  #8,8,walls

            ]


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

def get_args_json_file(path):
    with open(path + "/args.json", 'r') as f:
        args = json.load(f)
    f.close()
    return args

def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError()

def print_env(data, sizes, float_format=None):
    size_x, size_y = sizes[0], sizes[1]
    sys.stdout.write('-'*(size_x+2)+'\n')
    for h in range(size_y):
        sys.stdout.write('|')
        for w in range(size_x):
            if float_format:
                sys.stdout.write(float_format.format(data[xy_to_idx((w,h),size_x, size_y)]))
            else:
                sys.stdout.write(str(data[xy_to_idx((w,h), size_x, size_y)]))
        sys.stdout.write('|\n')
    sys.stdout.write('-' * (size_x + 2)+'\n')


if __name__ == "__main__":

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Get args file (assumes all experiments share the same arguments).
    args = get_args_json_file(DATA_FOLDER_PATH + EXPS_DATA[0]['id'])

    # Load optimal policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + VAL_ITER_DATA
    print(f"Opening experiment {VAL_ITER_DATA}")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        val_iter_data = json.load(f)
        val_iter_data = json.loads(val_iter_data)
        val_iter_data = val_iter_data[0]
    f.close()

    # Load and parse data.
    data = {}
    for exp in EXPS_DATA:

        # Open data.
        print(f"Opening experiment {exp['id']}")
        exp_path = DATA_FOLDER_PATH + exp['id']
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            exp_data = json.loads(exp_data)
        f.close()

        # Parse data for each train run.
        parsed_data = {}

        # episode_rewards field.
        parsed_data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data])

        # epsilon_values field.
        # e_concatenated = np.array([e['epsilon_values'] for e in exp_data])
        # e_concatenated = np.mean(e_concatenated, axis=0)
        # parsed_data['epsilon_values'] = e_concatenated

        # Q_vals field.
        q_concatenated = np.array([e['Q_vals'] for e in exp_data])
        parsed_data['Q_vals'] = np.mean(q_concatenated, axis=0)

        # max_Q_vals field.
        parsed_data['max_Q_vals'] = np.array([e['max_Q_vals'] for e in exp_data])

        # policy field.
        parsed_data['policies'] = np.array([e['policy'] for e in exp_data])

        # states_counts field.
        parsed_data['states_counts'] = np.array([e['states_counts'] for e in exp_data])

        data[exp['id']] = parsed_data

    """
        Print policies.
    """
    print(f'{VAL_ITER_DATA} policy:')
    print_env(val_iter_data['policy'], (args['env_args']['size_x'], args['env_args']['size_y']))
    for exp in EXPS_DATA:
        for (i, policy) in enumerate(data[exp['id']]['policies']):
            print(f"{exp['label']} policy (run {i}):")
            print_env(policy, (args['env_args']['size_x'], args['env_args']['size_y']))

    """
        Print max Q-values.
    """
    print('\n')
    print(f'{VAL_ITER_DATA} max Q-values:')
    print_env(val_iter_data['max_Q_vals'], (args['env_args']['size_x'], args['env_args']['size_y']), float_format="{:.1f} ")
    for exp in EXPS_DATA:
        for i, qs in enumerate(data[exp['id']]['max_Q_vals']):
            print(f"{exp['label']} max Q-values (run {i}):")
            print_env(qs, (args['env_args']['size_x'], args['env_args']['size_y']), float_format="{:.1f} ")

    """
        Print states counts.
    """
    print('\n')
    for exp in EXPS_DATA:
        for i, counts in enumerate(data[exp['id']]['states_counts']):
            print(f"{exp['label']} states_counts (run {i}):")
            print_env(counts, (args['env_args']['size_x'], args['env_args']['size_y']), float_format="{:.0f} ")

    """
        Plot episode rewards (averaged over all runs).
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp in EXPS_DATA:
        rewards = data[exp['id']]['episode_rewards']
        Y = np.mean(rewards, axis=0)
        X = np.linspace(1, len(Y), len(Y))
        lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

        p = plt.plot(X, Y, alpha=0.20)
        plt.plot(X, lowess[:,1], color=p[0].get_color(), label=exp['label'], zorder=10)

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.legend()

    plt.savefig('{0}/episode_rewards_avg.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards_avg.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Episode rewards (for each experiment and run).
    """
    num_cols = 3
    num_rows = math.ceil(len(EXPS_DATA) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    for (ax, exp) in zip(axs.flat, EXPS_DATA):

        for (i, run_data) in enumerate(data[exp['id']]['episode_rewards']):
            Y = run_data
            X = np.linspace(1, len(Y), len(Y))
            lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

            p = ax.plot(X, Y, alpha=0.20)
            ax.plot(X, lowess[:,1], color=p[0].get_color(), label=f"{i}", zorder=10)

        ax.legend()
        ax.set_ylabel('Reward')
        ax.set_xlabel('Episode')
        ax.set_title(exp['label'])

    plt.savefig('{0}/episode_rewards.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Epsilon.
    """
    """ fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp in EXPS_DATA:
        Y = data[exp['id']]['epsilon_values']
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X,Y,label=exp['label'])

    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.savefig('{0}/episode_epsilon.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_epsilon.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close() """

    """
        Q-values plots.
    """
    # Prepare data to plot.
    val_iter_Q_vals = np.array(val_iter_data['Q_vals'])
    exp_Q_vals = {}
    for exp in EXPS_DATA:
        exp_Q_vals[exp['id']] = np.array(data[exp['id']]['Q_vals'])

    # Sum plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp in EXPS_DATA:

        Y = np.sum(np.abs(val_iter_Q_vals - exp_Q_vals[exp['id']]), axis=(1,2))
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X, Y, label=exp['label'])

    plt.xlabel('Episode')
    plt.ylabel('Q-values error')
    plt.title('sum(abs(val_iter_Q_vals - Q_vals))')

    plt.legend()

    plt.savefig('{0}/q_values_summed_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_summed_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Mean + std/CI plot.
    num_cols = 3
    y_axis_range = [0,6]
    num_rows = math.ceil(len(EXPS_DATA) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    for (ax, exp) in zip(axs.flat, EXPS_DATA):

        Q_abs_diffs = np.abs(val_iter_Q_vals - exp_Q_vals[exp['id']])
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

        p = ax.plot(X,Y,label='Q-vals')
        ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)
        # ax.fill_between(X, Y-Y_std, Y+Y_std, color=p[0].get_color(), alpha=0.15)

        # Max Q-values.
        Q_abs_diffs = np.abs(np.max(val_iter_Q_vals, axis=1) - np.max(exp_Q_vals[exp['id']], axis=2))
        Y = np.mean(Q_abs_diffs, axis=1)
        X = np.linspace(1, len(Y), len(Y))
        # CI calculation.
        X_CI = np.arange(0, len(Y), 100)
        CI_bootstrap = [calculate_CI_bootstrap(Y[x], Q_abs_diffs[x,:]) for x in X_CI]
        CI_bootstrap = np.array(CI_bootstrap).T
        CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

        p = ax.plot(X, Y, label='Max Q-vals')
        ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)

        ax.legend()
        ax.set_ylim(y_axis_range)
        ax.set_ylabel('Q-values error')
        ax.set_xlabel('Episode')
        ax.set_title(exp['label'])

    fig.suptitle('mean(abs(val_iter_Q_vals - Q_vals))')

    plt.legend()

    plt.savefig('{0}/q_values_mean_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_mean_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Distribution plot.
    num_cols = 3
    y_axis_range = [0,11]
    num_rows = math.ceil(len(EXPS_DATA) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    val_iter_Q_vals_flattened = val_iter_Q_vals.flatten()

    for (ax, exp) in zip(axs.flat, EXPS_DATA):

        exp_Q_vals_flattened = exp_Q_vals[exp['id']].reshape(exp_Q_vals[exp['id']].shape[0], -1)

        abs_diff = np.abs(val_iter_Q_vals_flattened - exp_Q_vals_flattened)
        X = np.arange(0, len(abs_diff), 500)
        abs_diff_list = abs_diff[X,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(abs_diff[X,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, label=exp['label'], s=12)

        ax.set_ylim(y_axis_range)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-values error')
        ax.set_title(exp['label'])
        #ax.legend()

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals)')

    plt.savefig('{0}/q_values_violinplot_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_violinplot_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Distribution plot for max Q-values.
    num_cols = 3
    y_axis_range = [0,11]
    num_rows = math.ceil(len(EXPS_DATA) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    val_iter_Q_vals_flattened = np.max(val_iter_Q_vals, axis=1)

    for (ax, exp) in zip(axs.flat, EXPS_DATA):

        exp_Q_vals_flattened = np.max(exp_Q_vals[exp['id']], axis=2)

        abs_diff = np.abs(val_iter_Q_vals_flattened - exp_Q_vals_flattened)
        X = np.arange(0, len(abs_diff), 500)
        abs_diff_list = abs_diff[X,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(abs_diff[X,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, label=exp['label'], s=12)

        ax.set_ylim(y_axis_range)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-values error')
        ax.set_title(exp['label'])
        #ax.legend()

    fig.suptitle('Dist. of abs(max(val_iter_Q_vals) - max(Q_vals))')

    plt.legend()

    plt.savefig('{0}/q_values_violinplot_maxQ_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_violinplot_maxQ_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()
