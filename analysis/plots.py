import os
import sys
import math
import json
import random
import numpy as np
import pandas as pd
import pathlib
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import statsmodels.api as sm

from envs import env_suite

FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiments id.', required=True)
    parser.add_argument('--val_iter_exp', help='Value iteration experiments id.', required=True)

    return parser.parse_args()

# def calculate_CI_bootstrap(x_hat, samples, num_resamples=20000):
#     """
#         Calculates 95 % interval using bootstrap.
#         REF: https://ocw.mit.edu/courses/mathematics/
#             18-05-introduction-to-probability-and-statistics-spring-2014/
#             readings/MIT18_05S14_Reading24.pdf
#     """
#     resampled = np.random.choice(samples,
#                                 size=(len(samples), num_resamples),
#                                 replace=True)
#     means = np.mean(resampled, axis=0)
#     diffs = means - x_hat
#     bounds = [x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95)]
#
#     return bounds

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

# def ridgeline(x, data, ax, overlap=0, fill=True, labels=None, n_points=150):
#     if overlap > 1 or overlap < 0:
#         raise ValueError('overlap must be in [0 1]')
#     # xx = np.linspace(np.min(np.concatenate(data)),
#     #                  np.max(np.concatenate(data)), n_points)
#     # curves = []
#     # ys = []
#     for i, d in enumerate(data):
#         # pdf = gaussian_kde(d)
#         y = i*(1.0-overlap)
#         # ys.append(y)
#         # curve = pdf(xx)
#         # if fill:
#         #     ax.fill_between(xx, np.ones(n_points)*y, 
#         #                      curve+y, zorder=len(data)-i+1, color=fill)
#         ax.plot(x, d+y, c='k', zorder=len(data)-i+1)
#     # if labels:
#     #     ax.yticks(ys, labels)


def main(exp_id, val_iter_exp):

    # Parse arguments if needed.
    if (exp_id is None) or (val_iter_exp is None):
        c_args = get_arguments()
        exp_id = c_args.exp
        val_iter_exp = c_args.val_iter_exp

    print('Arguments (analysis/plots.py):')
    print('Exp. id: {0}'.format(exp_id))
    print('Val. iter. exp. id: {0}\n'.format(val_iter_exp))

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/'
    os.makedirs(output_folder, exist_ok=True)
    replay_buffer_plots_path = f'{output_folder}/replay_buffer'
    os.makedirs(replay_buffer_plots_path, exist_ok=True)
    q_vals_folder_path = PLOTS_FOLDER_PATH + exp_id + '/Q-values'
    os.makedirs(q_vals_folder_path, exist_ok=True)
    learner_plots_folder_path = PLOTS_FOLDER_PATH + exp_id + '/learner'
    os.makedirs(learner_plots_folder_path, exist_ok=True)

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

    # Open data.
    print(f"Opening experiment {exp_id}")
    exp_path = DATA_FOLDER_PATH + exp_id
    with open(exp_path + "/train_data.json", 'r') as f:
        exp_data = json.load(f)
        exp_data = json.loads(exp_data)
    f.close()

    # Parse data.
    data = {}

    # Q_vals_steps field.
    data['Q_vals_steps'] = exp_data[0]['Q_vals_steps'] # [(S)]

    # Q_vals field.
    data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,(S),S,A]

    # replay_buffer_counts_episodes field.
    data['replay_buffer_counts_steps'] = exp_data[0]['replay_buffer_counts_steps'] # [(S)]

    # replay_buffer_counts field.
    data['replay_buffer_counts'] = np.array([e['replay_buffer_counts'] for e in exp_data]) # [R,(S),S,A]

    # rollouts_steps field.
    data['rollouts_steps'] = exp_data[0]['rollouts_steps'] # [(S)]

    # rollouts_rewards field.
    data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(S),num_rollouts_types,num_rollouts]

    # Scalar metrics dict.
    scalar_metrics = {}

    is_grid_env = exp_args['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys()
    print('is_grid_env:', is_grid_env)
    if is_grid_env:
        lateral_size = int(np.sqrt(len(val_iter_data['policy']))) # Assumes env. is a square.

    # Load learner(s) CSV data (get all 'logs.csv' files from 'exp_path' folder).
    learner_csv_files = [str(p) for p in list(pathlib.Path(exp_path).rglob('logs.csv'))]
    print('Number of learner csv (logs.csv) files found: {0}'.format(len(learner_csv_files)))
    print(learner_csv_files)

    """
        `rollouts_rewards` plot.
    """
    if exp_args['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys():
        rollouts_types = sorted(env_suite.CUSTOM_GRID_ENVS[exp_args['env_name']].keys())
    elif exp_args['env_name'] == 'pendulum':
        rollouts_types = sorted(env_suite.PENDULUM_ENVS.keys())
    elif exp_args['env_name'] == 'mountaincar':
        rollouts_types = sorted(env_suite.MOUNTAINCAR_ENVS.keys())
    elif exp_args['env_name'] == 'multiPathsEnv':
        rollouts_types = sorted(env_suite.MULTIPATHS_ENVS.keys())
    else:
        raise ValueError(f'Env. {exp_args["env_name"]} does not have rollout types defined.')

    for t, rollout_type in enumerate(rollouts_types):
        print(f'Computing `rollouts_rewards_{rollout_type}` plot.')
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        rollout_type_data = data['rollouts_rewards'][:,:,t,:] # [R,(S),num_rollouts]

        for (i, run_rollouts) in enumerate(rollout_type_data): # run_rollouts = [(S),num_rollouts]
            X = data['rollouts_steps'] # [(S)]
            Y = np.mean(run_rollouts, axis=1) # [(S)]

            plt.plot(X, Y, label=f'Run {i}')

        plt.xlabel('Learning step')
        plt.ylabel('Rollouts average reward')
        plt.legend()

        plt.savefig('{0}/rollouts_rewards_{1}.png'.format(output_folder,rollout_type), bbox_inches='tight', pad_inches=0)
        plt.close()

    """
        Scalar evaluation rewards metrics.
    """
    for t, rollout_type in enumerate(rollouts_types):
        rollout_type_data = data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]
        scalar_metrics[f'rewards_{rollout_type}'] = np.mean(rollout_type_data[:,-10:,:])

    """
        `maximizing_action_s_*` plots.
    """
    if exp_args['env_name'] not in ('pendulum', 'mountaincar'):
        print('Computing `maximizing_action_s_*` plots.')

        for state in range(data['Q_vals'].shape[2]):

            num_cols = 3
            num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_data) in zip(axs.flat, data['Q_vals']): # run_data = [E,S,A]

                state_data = run_data[:,state,:] # [E,A]
                Y = np.argmax(state_data, axis=1) # [E]
                X = data['Q_vals_steps']

                ax.plot(X, Y)

                ax.set_ylim([-0.1,data['Q_vals'].shape[-1]+0.1])
                ax.set_ylabel('Maximizing action')
                ax.set_xlabel('Learning step')
                ax.set_title(f'Run {i}')

                i += 1

            if is_grid_env:
                fig.suptitle(f'Maximizing actions: state {state}; line {state // lateral_size }, col {state % lateral_size}')
            else:
                fig.suptitle(f'Maximizing actions: state {state}')

            plt.savefig(f'{q_vals_folder_path}/maximizing_action_s_{state}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    """
        `q_values_summed_error` plot.
    """
    print('Computing `q_values_summed_error` plot.')
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    
    for (i, run_Q_vals) in enumerate(data['Q_vals']):
        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        Y = np.sum(errors, axis=(1,2)) # [E]
        X = data['Q_vals_steps']

        plt.plot(X, Y, label=f'Run {i}')

    plt.xlabel('Learning step')
    plt.ylabel('Q-values error')
    plt.title('sum(abs(val_iter_Q_vals - Q_vals))')
    plt.legend()

    plt.savefig('{0}/q_values_summed_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Scalar q-values metrics.
    """
    errors = np.abs(val_iter_data['Q_vals'] - data['Q_vals']) # [R,E,S,A]
    errors = np.sum(errors, axis=(2,3)) # [R,E]
    scalar_metrics['qvals_summed_error'] = np.mean(errors[:,-10:])

    errors = np.abs(val_iter_data['Q_vals'] - data['Q_vals']) # [R,E,S,A]
    errors = np.mean(errors, axis=(2,3)) # [R,E]
    scalar_metrics['qvals_avg_error'] = np.mean(errors[:,-10:])

    """
        `q_values_mean_error` plot.
    """
    print('Computing `q_values_mean_error` plot.')
    num_cols = 3
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        Y = np.mean(errors, axis=(1,2)) # [E]
        X = data['Q_vals_steps']
        # Y_std = np.std(errors, axis=(1,2))

        # CI calculation.
        # errors_flatten = errors.reshape(len(Y), -1) # [E,S*A]
        # X_CI = np.arange(0, len(Y), 100)
        # CI_bootstrap = [calculate_CI_bootstrap(Y[x], errors_flatten[x,:]) for x in X_CI]
        # CI_bootstrap = np.array(CI_bootstrap).T
        # CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        # CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

        p = ax.plot(X,Y,label='Q-vals')
        # ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)
        # ax.fill_between(X, Y-Y_std, Y+Y_std, color=p[0].get_color(), alpha=0.15)

        # Max Q-values.
        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        maximizing_actions = np.argmax(run_Q_vals, axis=2) # [E,S]
        x, y = np.indices(maximizing_actions.shape)
        errors = errors[x,y,maximizing_actions] # [E,S]

        Y = np.mean(errors, axis=1) # [E]
        X = data['Q_vals_steps']
        # CI calculation.
        # X_CI = np.arange(0, len(Y), 100)
        # CI_bootstrap = [calculate_CI_bootstrap(Y[x], errors[x,:]) for x in X_CI]
        # CI_bootstrap = np.array(CI_bootstrap).T
        # CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        # CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

        p = ax.plot(X, Y, label='Max Q-vals')
        # ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)

        #ax.set_ylim(y_axis_range)
        ax.set_ylabel('Q-values error')
        ax.set_xlabel('Learning step')
        ax.set_title(f'Run {i}')

        i += 1

    plt.legend()

    plt.savefig('{0}/q_values_mean_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        `q_values_violinplot_error` plot.
    """
    print('Computing `q_values_violinplot_error` plot.')
    num_cols = 3
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [E,S,A]
        errors = errors.reshape(errors.shape[0], -1) # [E,S*A]

        X = np.arange(0, data['Q_vals_steps'][-1], 500)
        idxs = np.searchsorted(data['Q_vals_steps'], X)
        abs_diff_list = errors[idxs,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[idxs,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, s=12)

        #ax.set_ylim(y_axis_range)
        ax.set_xlabel('Learning step')
        ax.set_ylabel('Q-values error')
        ax.set_title(label=f"Run {i}")

        i += 1

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals)')

    plt.savefig('{0}/q_values_violinplot_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        `q_values_violinplot_maxQ_error` plot.
    """
    print('Computing `q_values_violinplot_maxQ_error` plot.')
    num_cols = 3
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

        X = np.arange(0, data['Q_vals_steps'][-1], 500)
        idxs = np.searchsorted(data['Q_vals_steps'], X)
        abs_diff_list = errors[idxs,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[idxs,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, s=12)

        #ax.set_ylim(y_axis_range)
        ax.set_xlabel('Learning step')
        ax.set_ylabel('Q-values error')
        ax.set_title(label=f"Run {i}")

        i += 1

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals) for a=argmax(Q_vals)')
    plt.legend()

    plt.savefig('{0}/q_values_violinplot_maxQ_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        `Q_values_s*-*-*` plots.
    """
    print('Computing `Q_values_s*-*-*` plots.')
    if exp_args['env_name'] not in ('pendulum', 'mountaincar'):
        for state_to_plot in range(data['Q_vals'].shape[2]):

            # Plot.
            num_cols = 3
            num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

                for action in range(run_Q_vals.shape[-1]):
                    
                    # Plot predicted Q-values.
                    Y = run_Q_vals[:,state_to_plot,action]
                    X = data['Q_vals_steps']
                    l = ax.plot(X, Y, label=f'Action {action}')

                    # Plot true Q-values.
                    ax.hlines(val_iter_data['Q_vals'][state_to_plot,action],
                            xmin=0, xmax=max(data['Q_vals_steps']), linestyles='--', color=l[0].get_color())

                #ax.set_ylim(y_axis_range)
                ax.set_xlabel('Learning step')
                ax.set_ylabel('Q-value')
                ax.set_title(label=f"Run {i}")
                ax.legend()

                i += 1

            if is_grid_env:
                fig.suptitle(f'Q-values: state {state}; line {state // lateral_size }, col {state % lateral_size}')
            else:
                fig.suptitle(f'Q-values: state {state}')

            plt.savefig(f'{q_vals_folder_path}/Q_values_s{state_to_plot}.png', bbox_inches='tight', pad_inches=0)

            plt.close()

    """
        Replay buffer statistics: P(a|s).
    """
    if exp_args['env_name'] not in ('pendulum', 'mountaincar'):
        print('Computing `Replay buffer: P(a|s)` plots.')

        for state in range(data['Q_vals'].shape[2]):

            num_cols = 3
            num_rows = math.ceil(data['replay_buffer_counts'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_data) in zip(axs.flat, data['replay_buffer_counts']): # run_data = [(E),S,A]

                state_data = run_data[:,state,:] # [(E),A]

                row_sums = np.sum(state_data, axis=1) # [(E)]
                action_probs = state_data / (row_sums[:, np.newaxis] + 1e-05) # [(E), A]

                for a in range(action_probs.shape[1]):
                    Y = action_probs[:,a]
                    X = data['replay_buffer_counts_steps']
                    ax.plot(X, Y, label=f'Action {a}')

                ax.set_ylim([0, 1.1])
                ax.set_ylabel('P(a|s)')
                ax.set_xlabel('Learning step')
                ax.set_title(f'Run {i}')

                ax.legend()

                i += 1

            if is_grid_env:
                fig.suptitle(f'Replay buffer: P(a|s): state {state}; line {state // lateral_size }, col {state % lateral_size}')
            else:
                fig.suptitle(f'Replay buffer: P(a|s): state {state}')

            plt.savefig(f'{replay_buffer_plots_path}/P_a_{state}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    """
        Replay buffer statistics: H[P(a|s)].
    """
    if exp_args['env_name'] not in ('pendulum', 'mountaincar'):
        print('Computing `Replay buffer: P(a|s) entropy` plots.')

        for state in range(data['Q_vals'].shape[2]):

            num_cols = 3
            num_rows = math.ceil(data['replay_buffer_counts'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_data) in zip(axs.flat, data['replay_buffer_counts']): # run_data = [(E),S,A]

                state_data = run_data[:,state,:] # [(E),A]

                row_sums = np.sum(state_data, axis=1) # [(E)]
                action_probs = state_data / (row_sums[:, np.newaxis] + 1e-05) # [(E), A]

                Y = -np.sum(action_probs * np.log(action_probs+1e-8), axis=1)
                X = data['replay_buffer_counts_steps']
                ax.plot(X, Y)

                #ax.set_ylim(y_axis_range)
                ax.set_ylabel('H[P(a|s)]')
                ax.set_xlabel('Learning step')
                ax.set_title(f'Run {i}')

                i += 1

            if is_grid_env:
                fig.suptitle(f'Replay buffer: H[P(a|s)]: state {state}; line {state // lateral_size }, col {state % lateral_size}')
            else:
                fig.suptitle(f'Replay buffer: H[P(a|s)]: state {state}')

            plt.savefig(f'{replay_buffer_plots_path}/P_a_entropy_{state}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    """
        Replay buffer statistics: H[P(s)].
    """
    print('Computing `Replay buffer: H[P(s)]` plots.')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for i, run_data in enumerate(data['replay_buffer_counts']): # run_data = [(E),S,A]

        aggregated_data = np.sum(run_data, axis=2) # [(E),S]
        row_sums = np.sum(aggregated_data, axis=1) # [(E)]
        state_probs = aggregated_data / (row_sums[:, np.newaxis] + 1e-05) # [(E), S]

        Y = -np.sum(state_probs * np.log(state_probs+1e-8), axis=1)
        X = data['replay_buffer_counts_steps'] 
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('H[P(s)]')
    plt.xlabel('Learning step')

    plt.legend()

    plt.title(f'Replay buffer: H[P(s)]')

    plt.savefig(f'{replay_buffer_plots_path}/P_s_entropy.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Replay buffer statistics: P(s).
    """
    """ print('Computing `Replay buffer: P(s)` plots.')

    num_cols = 3
    num_rows = math.ceil(data['replay_buffer_counts'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_data) in zip(axs.flat, data['replay_buffer_counts']): # run_data = [(E),S,A]

        aggregated_data = np.sum(run_data, axis=2) # [(E),S]
        row_sums = np.sum(aggregated_data, axis=1) # [(E)]
        state_probs = aggregated_data / row_sums[:, np.newaxis] # [(E), S]

        ridgeline(x=np.arange(0, state_probs.shape[1]), data=state_probs, ax=ax)
        # joypy.joyplot(state_probs.tolist(), ax=ax)

        ax.set_ylabel('P(s)')
        ax.set_xlabel('Episode')
        ax.set_title(f'Run {i}')

        ax.legend()

        i += 1

    fig.suptitle(f'Replay buffer: P(s)')

    plt.savefig(f'{replay_buffer_plots_path}/P_s.png', bbox_inches='tight', pad_inches=0)
    plt.close() """

    """
        Replay buffer statistics: H[P(s,a)].
    """
    print('Computing `Replay buffer: H[P(s,a)]` plot.')
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    final_entropies = []
    for i, run_data in enumerate(data['replay_buffer_counts']): # run_data = [(E),S,A]

        total_sum = np.sum(run_data, axis=(1,2)) # [(E)]
        sa_probs = run_data / (total_sum[:, np.newaxis, np.newaxis] + 1e-05) # [(E), S]

        Y = -np.sum(sa_probs * np.log(sa_probs+1e-8), axis=(1,2))
        X = data['replay_buffer_counts_steps']
        plt.plot(X, Y, label=f'Run {i}')

        final_entropies.append(np.mean(Y[-10:]))

    plt.ylabel('H[P(s,a)]')
    plt.xlabel('Learning step')

    plt.legend()

    plt.title(f'Replay buffer: H[P(s,a)]')

    plt.savefig(f'{replay_buffer_plots_path}/P_sa_entropy.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    scalar_metrics['replay_buffer_entropy'] = np.mean(final_entropies)

    """
        Replay buffer statistics: P(s) coverage.
    """
    print('Computing `Replay buffer: P(s) coverage` plot.')
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for i, run_data in enumerate(data['replay_buffer_counts']): # run_data = [(E),S,A]

        aggregated_data = np.sum(run_data, axis=2) # [(E),S]
        masked = (aggregated_data > 0)
        Y = np.sum(masked, axis=1) / masked.shape[1] # [(E)]
        X = data['replay_buffer_counts_steps']
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('Coverage')
    plt.xlabel('Learning step')
    plt.legend()

    plt.title(f'Replay buffer: P(s) coverage')

    plt.savefig(f'{replay_buffer_plots_path}/P_s_coverage.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Replay buffer statistics: P(s,a) coverage.
    """
    print('Computing `Replay buffer: P(s,a) coverage` plot.')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for i, run_data in enumerate(data['replay_buffer_counts']): # run_data = [(E),S,A]

        masked = (run_data > 0) # [(E),S,A]
        Y = np.sum(masked, axis=(1,2)) / (masked.shape[1]*masked.shape[2]) # [(E)]
        X = data['replay_buffer_counts_steps'] 
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('Coverage')
    plt.xlabel('Learning step')
    plt.legend()

    plt.title(f'Replay buffer: P(s,a) coverage')

    plt.savefig(f'{replay_buffer_plots_path}/P_s_a_coverage.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Learner(s): Loss(es) plots.
    """
    if len(learner_csv_files) > 0:

        learner_csv_cols = pd.read_csv(learner_csv_files[0]).columns

        for col in learner_csv_cols:

            if col in ('steps', 'walltime', 'step', 'learner_steps', 'learner_walltime'):
                continue

            print(f'Computing learner plot for column "{col}".')

            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            for learner_csv in learner_csv_files:
                df = pd.read_csv(learner_csv)
                plt.plot(df[col])

            plt.xlabel('Learner step')
            plt.ylabel(f'{col}')
            plt.savefig(learner_plots_folder_path + f"/{col}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

    # Store scalars dict.
    with open(output_folder + "scalar_metrics.json", 'w') as f:
        json.dump(scalar_metrics, f)
        f.close()

if __name__ == "__main__":
    main(exp_id=None, val_iter_exp=None)
