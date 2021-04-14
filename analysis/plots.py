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

# VAL_ITER_DATA = '8_8_val_iter_2021-04-13-17-55-56' # Env-8,8-1
VAL_ITER_DATA = '8_8_val_iter_2021-04-13-17-27-39' # Env-8,8-2
# VAL_ITER_DATA = '8_8_val_iter_2021-04-13-22-56-57' # Env-8,8-3

EXPS_DATA = [
            # {'id': '8_8_q_learning_2021-04-13-17-56-48', 'label': 'Q-learning'}, # Env-8,8-1

            #{'id': '8_8_dqn_2021-04-13-18-29-44', 'label': 'DQN+1hot+500k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-18-52-45', 'label': 'DQN+1hot+400k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-19-15-32', 'label': 'DQN+1hot+300k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-19-38-17', 'label': 'DQN+1hot+200k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-20-01-22', 'label': 'DQN+1hot+100k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-20-24-30', 'label': 'DQN+1hot+50k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-20-48-35', 'label': 'DQN+1hot+25k'}, # Env-8,8-1

            #{'id': '8_8_dqn_2021-04-13-21-12-55', 'label': 'DQN+smooth+500k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-21-36-27', 'label': 'DQN+smooth+400k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-21-59-39', 'label': 'DQN+smooth+300k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-22-22-29', 'label': 'DQN+smooth+200k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-22-45-18', 'label': 'DQN+smooth+100k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-23-08-11', 'label': 'DQN+smooth+50k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-13-23-31-12', 'label': 'DQN+smooth+25k'}, # Env-8,8-1

            #{'id': '8_8_dqn_2021-04-13-23-54-09', 'label': 'DQN+rand+500k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-14-00-16-45', 'label': 'DQN+rand+400k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-14-00-39-22', 'label': 'DQN+rand+300k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-14-01-02-03', 'label': 'DQN+rand+200k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-14-01-24-01', 'label': 'DQN+rand+100k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-14-01-45-49', 'label': 'DQN+rand+50k'}, # Env-8,8-1
            #{'id': '8_8_dqn_2021-04-14-02-07-55', 'label': 'DQN+rand+25k'}, # Env-8,8-1

            #################################################################################
            {'id': '8_8_q_learning_2021-04-13-22-33-17', 'label': 'Q-learning'}, # Env-8,8-2

            #################################################################################
            # {'id': '8_8_q_learning_2021-04-13-23-39-31', 'label': 'Q-learning'}, # Env-8,8-3 (20k episodes)

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
    val_iter_data['Q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]

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
        parsed_data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data]) # [R,E]

        # Q_vals field.
        parsed_data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]
        
        # max_Q_vals field.
        parsed_data['max_Q_vals'] = np.array([e['max_Q_vals'] for e in exp_data]) # [R,S]

        # policy field.
        parsed_data['policies'] = np.array([e['policy'] for e in exp_data]) # [R,S]

        # states_counts field.
        parsed_data['states_counts'] = np.array([e['states_counts'] for e in exp_data]) # [R,S]

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
        rewards = data[exp['id']]['episode_rewards'] # [R,E]
        Y = np.mean(rewards, axis=0) # [E]
        X = np.linspace(1, len(Y), len(Y))
        lowess = sm.nonparametric.lowess(Y, X, frac=0.10, is_sorted=True, it=0)

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
            Y = run_data # [E]
            X = np.linspace(1, len(Y), len(Y))
            lowess = sm.nonparametric.lowess(Y, X, frac=0.10, is_sorted=True, it=0)

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
        Q-values plots.
    """
    # Sum plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp in EXPS_DATA:
        errors = np.abs(val_iter_data['Q_vals'] - data[exp['id']]['Q_vals']) # [R,E,S,A]
        errors = np.mean(errors, axis=0) # [E,S,A]
        Y = np.sum(errors, axis=(1,2)) # [E]
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

        errors = np.abs(val_iter_data['Q_vals'] - data[exp['id']]['Q_vals']) # [R,E,S,A]
        errors = np.mean(errors, axis=0) # [E,S,A]
        Y = np.mean(errors, axis=(1,2)) # [E]
        X = np.linspace(1, len(Y), len(Y))
        # Y_std = np.std(errors, axis=(1,2))

        # CI calculation.
        errors_flatten = errors.reshape(len(Y), -1) # [E,S*A]
        X_CI = np.arange(0, len(Y), 100)
        CI_bootstrap = [calculate_CI_bootstrap(Y[x], errors_flatten[x,:]) for x in X_CI]
        CI_bootstrap = np.array(CI_bootstrap).T
        CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

        p = ax.plot(X,Y,label='Q-vals')
        ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)
        # ax.fill_between(X, Y-Y_std, Y+Y_std, color=p[0].get_color(), alpha=0.15)

        # Max Q-values.
        errors = np.abs(val_iter_data['Q_vals'] - data[exp['id']]['Q_vals']) # [R,E,S,A]
        maximizing_actions = np.argmax(data[exp['id']]['Q_vals'], axis=3) # [R,E,S]
        x, y, z = np.indices(maximizing_actions.shape)
        errors = errors[x,y,z,maximizing_actions] # [R,E,S]
        errors = np.mean(errors, axis=0) # [E,S]
        Y = np.mean(errors, axis=1) # [E]
        X = np.linspace(1, len(Y), len(Y))
        # CI calculation.
        X_CI = np.arange(0, len(Y), 100)
        CI_bootstrap = [calculate_CI_bootstrap(Y[x], errors[x,:]) for x in X_CI]
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

    for (ax, exp) in zip(axs.flat, EXPS_DATA):

        errors = np.abs(val_iter_data['Q_vals'] - data[exp['id']]['Q_vals']) # [R,E,S,A]
        errors = np.mean(errors, axis=0) # [E,S,A]
        errors = errors.reshape(errors.shape[0], -1) # [E,S*A]

        X = np.arange(0, len(errors), 500)
        abs_diff_list = errors[X,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[X,:], axis=1) # Sub-sample.

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

    for (ax, exp) in zip(axs.flat, EXPS_DATA):

        errors = np.abs(val_iter_data['Q_vals'] - data[exp['id']]['Q_vals']) # [R,E,S,A]
        maximizing_actions = np.argmax(data[exp['id']]['Q_vals'], axis=3) # [R,E,S]
        x, y, z = np.indices(maximizing_actions.shape)
        errors = errors[x,y,z,maximizing_actions] # [R,E,S]
        errors = np.mean(errors, axis=0) # [E,S]

        X = np.arange(0, len(errors), 500)
        abs_diff_list = errors[X,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[X,:], axis=1) # Sub-sample.

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
