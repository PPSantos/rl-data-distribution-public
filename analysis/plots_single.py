import os
import sys
import math
import json
import random
import numpy as np
import pathlib
import argparse

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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiments id.', required=True)
    parser.add_argument('--val_iter_exp', help='Value iteration experiments id.', required=True)

    return parser.parse_args()

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


def main(exp_id, val_iter_exp):

    # Parse arguments if needed.
    if (exp_id is None) or (val_iter_exp is None):
        c_args = get_arguments()
        exp_id = c_args.exp
        val_iter_exp = args.val_iter_exp
        
    print('Arguments (analysis/plots_single.py):')
    print('\tExp. id: {0}'.format(exp_id))
    print('\tVal. iter. exp. id: {0}\n'.format(val_iter_exp))

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/'
    os.makedirs(output_folder, exist_ok=True)

    # Get args file (assumes all experiments share the same arguments).
    exp_args = get_args_json_file(DATA_FOLDER_PATH + exp_id)
    print('Exp. args:')
    print(exp_args)

    # Store a copy of the args.json file inside plots folder.
    with open(output_folder + "args.json", 'w') as f:
        json.dump(exp_args, f)
        f.close()

    # Load optimal policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + val_iter_exp
    print(f"Opening experiment {val_iter_exp}")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        val_iter_data = json.load(f)
        val_iter_data = json.loads(val_iter_data)
        val_iter_data = val_iter_data[0]
    f.close()
    val_iter_data['Q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]

    # Load and parse data.
    data = {}

    # Open data.
    print(f"Opening experiment {exp_id}")
    exp_path = DATA_FOLDER_PATH + exp_id
    with open(exp_path + "/train_data.json", 'r') as f:
        exp_data = json.load(f)
        exp_data = json.loads(exp_data)
    f.close()

    # Parse data for each train run.
    data = {}

    # episode_rewards field.
    data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data]) # [R,E]

    # Q_vals field.
    data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]
    
    # max_Q_vals field.
    data['max_Q_vals'] = np.array([e['max_Q_vals'] for e in exp_data]) # [R,S]

    # policy field.
    data['policies'] = np.array([e['policy'] for e in exp_data]) # [R,S]

    # states_counts field.
    data['states_counts'] = np.array([e['states_counts'] for e in exp_data]) # [R,S]


    """
        Print policies.
    """
    print(f'{val_iter_exp} policy:')
    print_env(val_iter_data['policy'], (exp_args['env_args']['size_x'], exp_args['env_args']['size_y']))
    for (i, policy) in enumerate(data['policies']):
        print(f"{exp_id} policy (run {i}):")
        print_env(policy, (exp_args['env_args']['size_x'], exp_args['env_args']['size_y']))

    """
        Print max Q-values.
    """
    print('\n')
    print(f'{val_iter_exp} max Q-values:')
    print_env(val_iter_data['max_Q_vals'], (exp_args['env_args']['size_x'], exp_args['env_args']['size_y']), float_format="{:.1f} ")
    for i, qs in enumerate(data['max_Q_vals']):
        print(f"{exp_id} max Q-values (run {i}):")
        print_env(qs, (exp_args['env_args']['size_x'], exp_args['env_args']['size_y']), float_format="{:.1f} ")

    """
        Print states counts.
    """
    print('\n')
    for i, counts in enumerate(data['states_counts']):
        print(f"{exp_id} states_counts (run {i}):")
        print_env(counts, (exp_args['env_args']['size_x'], exp_args['env_args']['size_y']), float_format="{:.0f} ")

    """
        Plot episode rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (i, run_rewards) in enumerate(data['episode_rewards']):
        Y = run_rewards # [E]
        X = np.linspace(1, len(Y), len(Y))
        lowess = sm.nonparametric.lowess(Y, X, frac=0.10, is_sorted=True, it=0)

        p = plt.plot(X, Y, alpha=0.20)
        plt.plot(X, lowess[:,1], color=p[0].get_color(), label=f'Run {i}', zorder=10)

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.legend()

    plt.savefig('{0}/episode_rewards_avg.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards_avg.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Q-values plots.
    """
    # Sum plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    
    for (i, run_Q_vals) in enumerate(data['Q_vals']):
        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        Y = np.sum(errors, axis=(1,2)) # [E]
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X, Y, label=f'Run {i}')

    plt.xlabel('Episode')
    plt.ylabel('Q-values error')
    plt.title('sum(abs(val_iter_Q_vals - Q_vals))')

    plt.legend()

    plt.savefig('{0}/q_values_summed_error.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_summed_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Mean + std/CI plot.
    num_cols = 3
    y_axis_range = [0,6]
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
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
        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        maximizing_actions = np.argmax(run_Q_vals, axis=2) # [E,S]
        x, y = np.indices(maximizing_actions.shape)
        errors = errors[x,y,maximizing_actions] # [E,S]

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

        ax.set_ylim(y_axis_range)
        ax.set_ylabel('Q-values error')
        ax.set_xlabel('Episode')
        ax.set_title(f'Run {i}')

        i += 1

    plt.legend()

    plt.savefig('{0}/q_values_mean_error.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_mean_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Distribution plot.
    num_cols = 3
    y_axis_range = [0,11]
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        errors = errors.reshape(errors.shape[0], -1) # [E,S*A]

        X = np.arange(0, len(errors), 500)
        abs_diff_list = errors[X,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[X,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, s=12)

        ax.set_ylim(y_axis_range)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-values error')
        ax.set_title(label=f"Run {i}")

        i += 1

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals)')

    plt.savefig('{0}/q_values_violinplot_error.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_violinplot_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Distribution plot for max Q-values.
    num_cols = 3
    y_axis_range = [0,11]
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        maximizing_actions = np.argmax(run_Q_vals, axis=2) # [E,S]
        x, y = np.indices(maximizing_actions.shape)
        errors = errors[x,y,maximizing_actions] # [E,S]

        X = np.arange(0, len(errors), 500)
        abs_diff_list = errors[X,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[X,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, s=12)

        ax.set_ylim(y_axis_range)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-values error')
        ax.set_title(label=f"Run {i}")

        i += 1

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals) for a=argmax(Q_vals)')

    plt.legend()

    plt.savefig('{0}/q_values_violinplot_maxQ_error.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_violinplot_maxQ_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main(exp_id=None,val_iter_exp=None)