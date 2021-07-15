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

    print('Arguments (analysis/plots_single.py):')
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

    # episode_rewards field.
    data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data]) # [R,E]

    # Q_vals_episodes field.
    data['Q_vals_episodes'] = exp_data[0]['Q_vals_episodes'] # [(E)]

    # Q_vals field.
    data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,(E),S,A]
    
    # max_Q_vals field.
    data['max_Q_vals'] = np.array([e['max_Q_vals'] for e in exp_data]) # [R,S]

    # policy field.
    data['policies'] = np.array([e['policy'] for e in exp_data]) # [R,S]

    # states_counts field.
    data['states_counts'] = np.array([e['states_counts'] for e in exp_data]) # [R,S]

    # replay_buffer_counts_episodes field.
    data['replay_buffer_counts_episodes'] = exp_data[0]['replay_buffer_counts_episodes'] # [(E)]

    # replay_buffer_counts field.
    data['replay_buffer_counts'] = np.array([e['replay_buffer_counts'] for e in exp_data]) # [R,(E),S,A]

    # rollouts_episodes field.
    data['rollouts_episodes'] = exp_data[0]['rollouts_episodes'] # [(E)]

    # rollouts_rewards field.
    data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]

    """# e_losses field.
    data['e_losses'] = np.array([e['e_losses'] for e in exp_data]) # [R,(E),E_vals_learning_iters]

    # E_vals field.
    data['E_vals'] = np.array([e['E_vals'] for e in exp_data]) # [R,(E),S,A]

    # Q_errors field.
    if 'Q_errors' in exp_data[0].keys():
        data['Q_errors'] = np.array([e['Q_errors'] for e in exp_data]) # [R,(E),S,A]

    data['E_vals'] = np.array([e['E_vals'] for e in exp_data]) # [R,(E),S,A]"""

    # Scalar metrics dict.
    scalar_metrics = {}

    is_grid_env = exp_args['env_args']['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys()
    print('is_grid_env:', is_grid_env)
    if is_grid_env:
        lateral_size = int(np.sqrt(len(val_iter_data['policy']))) # Assumes env. is a square.

    # Load learner(s) CSV data (get all 'logs.csv' files from 'exp_path' folder).
    learner_csv_files = [str(p) for p in list(pathlib.Path(exp_path).rglob('logs.csv'))]
    print('Number of learner csv (logs.csv) files found: {0}'.format(len(learner_csv_files)))
    print(learner_csv_files)

    ################################################
    """to_plot_idx = -5

    print(f"Episode = {data['Q_vals_episodes'][to_plot_idx]}")

    N_runs = data['E_vals'].shape[0]
    N_states = data['E_vals'].shape[2]

    for run in range(N_runs):
        print('='*40)
        print('='*40)
        print(f'Run {run}:')
        Q_vals_run = data['Q_vals'][run][to_plot_idx] # [S,A]
        E_vals_run = data['E_vals'][run][to_plot_idx] # [S,A]

        print('Max Q-vals:')
        max_q_vals = [np.max(Q_vals_run[s,:]) for s in range(N_states)]
        print_env(max_q_vals, (lateral_size, lateral_size), float_format="{:.1f} ")

        print('Q-vals max policy:')
        max_policy = [np.argmax(Q_vals_run[s,:]) for s in range(N_states)] # S
        print_env(max_policy, (lateral_size, lateral_size))

        print('|Q-Q*| (mean error):')
        sa_errors = np.abs(val_iter_data['Q_vals'] - Q_vals_run) # [S,A]
        errors_states = np.mean(sa_errors, axis=1)
        print_env(errors_states, (lateral_size, lateral_size), float_format="{:.1f} ")

        print('|Q-Q*| (max error):')
        sa_errors = np.abs(val_iter_data['Q_vals'] - Q_vals_run) # [S,A]
        errors_states = np.max(sa_errors, axis=1)
        print_env(errors_states, (lateral_size, lateral_size), float_format="{:.1f} ")

        print('|Q-Q*| (error for a=argmax(Q)):')
        sa_errors = np.abs(val_iter_data['Q_vals'] - Q_vals_run) # [S,A]
        argmax_errors = [sa_errors[s,a_max] for (s, a_max) in enumerate(max_policy)] # S
        print_env(argmax_errors, (lateral_size, lateral_size), float_format="{:.1f} ")

        print('Max E-vals:')
        max_e_vals = [np.max(E_vals_run[s,:]) for s in range(N_states)]
        print_env(max_e_vals, (lateral_size, lateral_size), float_format="{:.1f} ")

        print('E-vals max policy:')
        max_policy = [np.argmax(E_vals_run[s,:]) for s in range(N_states)] # S
        print_env(max_policy, (lateral_size, lateral_size))

        # E-values learning losses.
        # fig = plt.figure()
        # fig.set_size_inches(FIGURE_X, FIGURE_Y)
        # plt.plot(data['e_losses'][run, to_plot_idx,:])
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

    
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    to_plot_idxs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]
    #to_plot_idxs = [5,6,7,8,9,10,11,12,14,15]

    q_vals_summed = np.zeros((64,5)) # [S,A]
    max_q_vals = []
    max_e_vals = []
    errors = []
    buffer_counts = []
    Q_errors = []

    print(data['Q_vals'].shape)
    print(data['E_vals'].shape)
    print(data['replay_buffer_counts'].shape)
    print('-'*30)

    for run in range(N_runs):
        for _idx in to_plot_idxs:

            Q_vals_run = data['Q_vals'][run][_idx] # [S,A]
            E_vals_run = data['E_vals'][run][_idx] # [S,A]
            replay_counts_run = data['replay_buffer_counts'][run][_idx] # [S,A]
            if 'Q_errors' in exp_data[0].keys():
                Q_errors_run = data['Q_errors'][run][_idx] # [S,A]
                Q_errors.append(np.mean(Q_errors_run, axis=1)) # [S]

            max_e_vals.append([np.max(E_vals_run[s,:]) for s in range(N_states)])
            max_q_vals.append([np.max(Q_vals_run[s,:]) for s in range(N_states)])

            sa_errors = np.abs(val_iter_data['Q_vals'] - Q_vals_run) # [S,A]
            errors.append(np.mean(sa_errors, axis=1)) # [S]

            buffer_counts.append([np.sum(replay_counts_run[s,:]) for s in range(N_states)])

            q_vals_summed = q_vals_summed + Q_vals_run # [S,A]


    max_e_vals = np.array(max_e_vals)
    max_q_vals = np.array(max_q_vals)
    errors = np.array(errors)
    buffer_counts = np.array(buffer_counts)
    print('max_e_vals shape', max_e_vals.shape)
    print('max_q_vals shape', max_q_vals.shape)
    print('errors shape', errors.shape)
    print('buffer_counts shape', buffer_counts.shape)
    if 'Q_errors' in exp_data[0].keys():
        Q_errors = np.array(Q_errors)
        print('Q_errors shape', Q_errors.shape)"""

    """
        gridEnv1
    """
    """# max_q_vals
    max_q_vals = np.mean(max_q_vals, axis=0)
    max_q_vals = np.reshape(max_q_vals, (8,-1))
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(max_q_vals, linewidth=0.5, cmap="coolwarm", cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('max_q_vals.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('max_q_vals.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Greedy Q-vals policy
    q_vals_greedy_policy = np.argmax(q_vals_summed, axis=1) # [S]
    q_vals_greedy_policy = np.reshape(q_vals_greedy_policy, (8,-1))

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(q_vals_greedy_policy, linewidth=0.5, cmap="coolwarm")#, cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    #plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
    #plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('q_vals_greedy_policy.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('q_vals_greedy_policy.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # max_e_vals
    max_e_vals = np.mean(max_e_vals, axis=0)
    max_e_vals = np.reshape(max_e_vals, (8,-1))
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(max_e_vals, linewidth=0.5, cmap="coolwarm", cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('max_e_vals.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('max_e_vals.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # buffer_counts
    buffer_counts = np.mean(buffer_counts, axis=0)
    buffer_counts = np.reshape(buffer_counts, (8,-1))
    buffer_counts[0,7] = np.nan
    buffer_counts[7,0] = np.nan
    mask_array = np.ma.masked_invalid(buffer_counts).mask
    labels = buffer_counts / np.nansum(buffer_counts) * 100
    labels = np.around(labels, decimals=1)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(buffer_counts, annot=labels, linewidth=0.5, cmap="coolwarm", cbar=False, mask=mask_array)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    #plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
    #plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('buffer_counts.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('buffer_counts.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # errors: abs(Q_phi - Q*)
    errors = np.mean(errors, axis=0)
    errors = np.reshape(errors, (8,-1))
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(errors, linewidth=0.5, cmap="coolwarm", cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('errors.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('errors.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    if 'Q_errors' in exp_data[0].keys():
        # errors: abs(Q_phi - (r+gamma*max(Q[s_t1,:])))

        Q_errors = np.mean(Q_errors, axis=0)
        Q_errors = np.reshape(Q_errors, (8,-1))
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(Q_errors, linewidth=0.5, cmap="coolwarm", cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig('Q_errors.png', bbox_inches='tight', pad_inches=0)
        plt.savefig('Q_errors.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()"""

    """
        gridEnv4
    """
    """colormap = sns.color_palette("coolwarm", as_cmap=True)
    colormap.set_bad("black")
    # max_q_vals
    max_q_vals = np.mean(max_q_vals, axis=0)
    max_q_vals = np.reshape(max_q_vals, (8,-1))
    max_q_vals = np.delete(max_q_vals, 0, 0)
    max_q_vals[3,[1,2,3,4,5,6]] = np.nan
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(max_q_vals, linewidth=0.5, cmap=colormap, cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('max_q_vals.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('max_q_vals.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Greedy Q-vals policy
    # q_vals_greedy_policy = np.argmax(q_vals_summed, axis=1) # [S]
    # q_vals_greedy_policy = np.reshape(q_vals_greedy_policy, (8,-1))

    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # sns.heatmap(q_vals_greedy_policy, linewidth=0.5, cmap="coolwarm")#, cbar=False)

    # plt.xticks([]) # remove the tick marks by setting to an empty list
    # plt.yticks([]) # remove the tick marks by setting to an empty list
    # plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    # plt.grid()
    # plt.savefig('q_vals_greedy_policy.pdf', bbox_inches='tight', pad_inches=0)
    # plt.savefig('q_vals_greedy_policy.png', bbox_inches='tight', pad_inches=0)
    # plt.close()

    # max_e_vals
    max_e_vals = np.mean(max_e_vals, axis=0)
    max_e_vals = np.reshape(max_e_vals, (8,-1))
    max_e_vals = np.delete(max_e_vals, 0, 0)
    max_e_vals[3,[1,2,3,4,5,6]] = np.nan
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(max_e_vals, linewidth=0.5, cmap=colormap, cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('max_e_vals.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('max_e_vals.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # buffer_counts
    buffer_counts = np.mean(buffer_counts, axis=0)
    buffer_counts = np.reshape(buffer_counts, (8,-1))
    buffer_counts = np.delete(buffer_counts, 0, 0)
    buffer_counts[3,[1,2,3,4,5,6]] = np.nan
    #buffer_counts[3,0] = np.nan # start
    #buffer_counts[3,7] = np.nan # goal
    mask_array = np.ma.masked_invalid(buffer_counts).mask
    labels = buffer_counts / np.nansum(buffer_counts) * 100
    labels = np.around(labels, decimals=1)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(buffer_counts, annot=labels, linewidth=0.5, cmap="coolwarm", cbar=False, mask=mask_array)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    #plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
    #plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('buffer_counts.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('buffer_counts.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # errors
    errors = np.mean(errors, axis=0)
    errors = np.reshape(errors, (8,-1))
    errors = np.delete(errors, 0, 0)
    errors[3,[1,2,3,4,5,6]] = np.nan
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.heatmap(errors, linewidth=0.5, cmap="coolwarm", cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig('errors.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('errors.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


    if 'Q_errors' in exp_data[0].keys():
        # errors: abs(Q_phi - (r+gamma*max(Q[s_t1,:])))

        Q_errors = np.mean(Q_errors, axis=0)
        Q_errors = np.reshape(Q_errors, (8,-1))
        Q_errors = np.delete(Q_errors, 0, 0)
        Q_errors[3,[1,2,3,4,5,6]] = np.nan

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(Q_errors, linewidth=0.5, cmap="coolwarm", cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig('Q_errors.png', bbox_inches='tight', pad_inches=0)
        plt.savefig('Q_errors.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

    exit()"""
    
    
    ################################################

    """
        Print policies.
    """
    if is_grid_env:
        print(f'{val_iter_exp} policy:')
        print_env(val_iter_data['policy'], (lateral_size, lateral_size))
        for (i, policy) in enumerate(data['policies']):
            print(f"{exp_id} policy (run {i}):")
            print_env(policy, (lateral_size, lateral_size))

    """
        Print max Q-values.
    """
    if is_grid_env:
        print('\n')
        print(f'{val_iter_exp} max Q-values:')
        print_env(val_iter_data['max_Q_vals'], (lateral_size, lateral_size), float_format="{:.1f} ")
        for i, qs in enumerate(data['max_Q_vals']):
            print(f"{exp_id} max Q-values (run {i}):")
            print_env(qs, (lateral_size, lateral_size), float_format="{:.1f} ")

    """
        Print states counts.
    """
    if is_grid_env:
        print('\n')
        for i, counts in enumerate(data['states_counts']):
            print(f"{exp_id} states_counts (run {i}):")
            print_env(counts, (lateral_size, lateral_size), float_format="{:.0f} ")

    """
        `episode_rewards_avg` plot.
    """
    print('Computing `episode_rewards_avg` plot.')
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

    plt.savefig('{0}/episode_rewards_avg.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Scalar train rewards metrics.
    """
    scalar_metrics['train_rewards_total'] = np.mean(data['episode_rewards'])

    """
        `rollouts_rewards` plot.
    """
    for t, rollout_type in enumerate(exp_args['rollouts_types']):
        print(f'Computing `rollouts_rewards_{rollout_type}` plot.')
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        rollout_type_data = data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]

        for (i, run_rollouts) in enumerate(rollout_type_data): # run_rollouts = [(E),num_rollouts]
            X = data['rollouts_episodes'] # [(E)]
            Y = np.mean(run_rollouts, axis=1) # [(E)]

            plt.plot(X, Y, label=f'Run {i}')

        plt.xlabel('Episode')
        plt.ylabel('Rollouts average reward')
        plt.legend()

        plt.savefig('{0}/rollouts_rewards_{1}.png'.format(output_folder,rollout_type), bbox_inches='tight', pad_inches=0)
        plt.close()

    """
        Scalar evaluation rewards metrics.
    """
    for t, rollout_type in enumerate(exp_args['rollouts_types']):
        rollout_type_data = data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]
        scalar_metrics[f'eval_rewards_total_{rollout_type}'] = np.mean(rollout_type_data)
        scalar_metrics[f'eval_rewards_final_{rollout_type}'] = np.mean(rollout_type_data[:,-1,:])

    """
        `maximizing_action_s_*` plots.
    """
    if exp_args['env_args']['env_name'] not in ('pendulum', 'mountaincar'):
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
                X = data['Q_vals_episodes']

                ax.plot(X, Y)

                ax.set_ylim([-0.1,data['Q_vals'].shape[-1]+0.1])
                ax.set_ylabel('Maximizing action')
                ax.set_xlabel('Episode')
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
        X = data['Q_vals_episodes']

        plt.plot(X, Y, label=f'Run {i}')

    plt.xlabel('Episode')
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
    scalar_metrics['train_qvals_summed_error_total'] = np.mean(errors)
    scalar_metrics['train_qvals_summed_error_final'] = np.mean(errors[:,-1])

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
        X = data['Q_vals_episodes']
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
        X = data['Q_vals_episodes']
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
        ax.set_xlabel('Episode')
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

        X = np.arange(0, data['Q_vals_episodes'][-1], 500)
        idxs = np.searchsorted(data['Q_vals_episodes'], X)
        abs_diff_list = errors[idxs,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[idxs,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, s=12)

        #ax.set_ylim(y_axis_range)
        ax.set_xlabel('Episode')
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

        X = np.arange(0, data['Q_vals_episodes'][-1], 500)
        idxs = np.searchsorted(data['Q_vals_episodes'], X)
        abs_diff_list = errors[idxs,:].tolist() # Sub-sample.
        abs_diff_mean = np.mean(errors[idxs,:], axis=1) # Sub-sample.

        violin = ax.violinplot(abs_diff_list, positions=X,
                                showextrema=True, widths=350)

        ax.scatter(X, abs_diff_mean, s=12)

        #ax.set_ylim(y_axis_range)
        ax.set_xlabel('Episode')
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
    if exp_args['env_args']['env_name'] not in ('pendulum', 'mountaincar'):
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
                    X = data['Q_vals_episodes']
                    l = ax.plot(X, Y, label=f'Action {action}')

                    # Plot true Q-values.
                    ax.hlines(val_iter_data['Q_vals'][state_to_plot,action],
                            xmin=0, xmax=max(data['Q_vals_episodes']), linestyles='--', color=l[0].get_color())

                #ax.set_ylim(y_axis_range)
                ax.set_xlabel('Episode')
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
    if exp_args['env_args']['env_name'] not in ('pendulum', 'mountaincar'):
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
                    X = data['replay_buffer_counts_episodes']
                    ax.plot(X, Y, label=f'Action {a}')

                ax.set_ylim([0, 1.1])
                ax.set_ylabel('P(a|s)')
                ax.set_xlabel('Episode')
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
    if exp_args['env_args']['env_name'] not in ('pendulum', 'mountaincar'):
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
                X = data['replay_buffer_counts_episodes']
                ax.plot(X, Y)

                #ax.set_ylim(y_axis_range)
                ax.set_ylabel('H[P(a|s)]')
                ax.set_xlabel('Episode')
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
        X = data['replay_buffer_counts_episodes'] 
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('H[P(s)]')
    plt.xlabel('Episode')

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

    for i, run_data in enumerate(data['replay_buffer_counts']): # run_data = [(E),S,A]

        total_sum = np.sum(run_data, axis=(1,2)) # [(E)]
        sa_probs = run_data / (total_sum[:, np.newaxis, np.newaxis] + 1e-05) # [(E), S]

        Y = -np.sum(sa_probs * np.log(sa_probs+1e-8), axis=(1,2))
        X = data['replay_buffer_counts_episodes']
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('H[P(s,a)]')
    plt.xlabel('Episode')

    plt.legend()

    plt.title(f'Replay buffer: H[P(s,a)]')

    plt.savefig(f'{replay_buffer_plots_path}/P_sa_entropy.png', bbox_inches='tight', pad_inches=0)
    plt.close()

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
        X = data['replay_buffer_counts_episodes']
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('Coverage')
    plt.xlabel('Episode')
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
        X = data['replay_buffer_counts_episodes'] 
        plt.plot(X, Y, label=f'Run {i}')

    plt.ylabel('Coverage')
    plt.xlabel('Episode')
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
    main(exp_id=None,val_iter_exp=None)