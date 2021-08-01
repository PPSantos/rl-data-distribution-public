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
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

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
        val_iter_exp = c_args.val_iter_exp

    print('Arguments (analysis/plots_e_vals.py):')
    print('Exp. id: {0}'.format(exp_id))
    print('Val. iter. exp. id: {0}\n'.format(val_iter_exp))

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/'
    os.makedirs(output_folder, exist_ok=True)
    e_vals_folder_path = PLOTS_FOLDER_PATH + exp_id + '/plots_e_vals'
    os.makedirs(e_vals_folder_path, exist_ok=True)

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

    # Q_vals_episodes field.
    data['Q_vals_episodes'] = exp_data[0]['Q_vals_episodes'] # [(E)]

    # Q_vals field.
    data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,(E),S,A]

    # replay_buffer_counts field.
    data['replay_buffer_counts'] = np.array([e['replay_buffer_counts'] for e in exp_data]) # [R,(E),S,A]

    # E_vals field.
    data['E_vals'] = np.array([e['E_vals'] for e in exp_data]) # [R,(E),S,A]

    # Q_errors field.
    if 'Q_errors' in exp_data[0].keys():
        data['Q_errors'] = np.array([e['Q_errors'] for e in exp_data]) # [R,(E),S,A]

    # E_vals field.
    data['E_vals'] = np.array([e['E_vals'] for e in exp_data]) # [R,(E),S,A]
    
    # estimated_E_vals field (the E-values estimated via the tabular E-learning algorithm).
    data['estimated_E_vals'] = np.array([e['estimated_E_vals'] for e in exp_data]) # [R,(E),S,A]

    is_grid_env = exp_args['env_args']['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys()
    print('is_grid_env:', is_grid_env)
    if is_grid_env:
        lateral_size = int(np.sqrt(len(val_iter_data['policy']))) # Assumes env. is a square.

    # Load learner(s) CSV data (get all 'logs.csv' files from 'exp_path' folder).
    learner_csv_files = [str(p) for p in list(pathlib.Path(exp_path).rglob('logs.csv'))]
    print('Number of learner csv (logs.csv) files found: {0}'.format(len(learner_csv_files)))
    print(learner_csv_files)

    print('data[Q_vals] shape:', data['Q_vals'].shape)
    print('data[E_vals] shape:', data['E_vals'].shape)
    print('data[replay_buffer_counts] shape:', data['replay_buffer_counts'].shape)

    N_runs, N_eps, N_states, N_actions = data['Q_vals'].shape
    print('N_runs:', N_runs)
    print('N_eps:', N_eps)
    print('N_states:', N_states)
    print('N_actions:', N_actions)

    """to_plot_idx = -5
    print('to_plot_idx:', to_plot_idx)
    print(f"Episode = {data['Q_vals_episodes'][to_plot_idx]}")


    for run in range(N_runs):
        print('='*30)
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
        print_env(max_policy, (lateral_size, lateral_size))"""
    
    #to_plot_idxs = [5,6,7,8,9,10,11,12,14,15]
    to_plot_idxs = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]

    q_vals_summed = np.zeros((N_states,N_actions)) # [S,A]
    e_vals_summed = np.zeros((N_states,N_actions)) # [S,A]
    max_q_vals = []
    max_e_vals = []
    max_estimated_e_vals = []
    errors = []
    buffer_counts = []
    Q_errors = []

    for run in range(N_runs):
        for _idx in to_plot_idxs:

            Q_vals_run = data['Q_vals'][run][_idx] # [S,A]
            E_vals_run = data['E_vals'][run][_idx] # [S,A]
            estimated_E_vals_run = data['estimated_E_vals'][run][_idx] # [S,A]
            replay_counts_run = data['replay_buffer_counts'][run][_idx] # [S,A]
            if 'Q_errors' in exp_data[0].keys():
                Q_errors_run = data['Q_errors'][run][_idx] # [S,A]
                Q_errors.append(np.mean(Q_errors_run, axis=1)) # [S]

            max_e_vals.append([np.max(E_vals_run[s,:]) for s in range(N_states)])
            max_estimated_e_vals.append([np.max(estimated_E_vals_run[s,:]) for s in range(N_states)])
            max_q_vals.append([np.max(Q_vals_run[s,:]) for s in range(N_states)])

            sa_errors = np.abs(val_iter_data['Q_vals'] - Q_vals_run) # [S,A]
            errors.append(np.mean(sa_errors, axis=1)) # [S]

            buffer_counts.append([np.sum(replay_counts_run[s,:]) for s in range(N_states)])

            q_vals_summed = q_vals_summed + Q_vals_run # [S,A]
            e_vals_summed = e_vals_summed + E_vals_run # [S,A]

    max_e_vals = np.array(max_e_vals)
    max_estimated_e_vals = np.array(max_estimated_e_vals)
    max_q_vals = np.array(max_q_vals)
    errors = np.array(errors)
    buffer_counts = np.array(buffer_counts)

    print('max_e_vals shape', max_e_vals.shape)
    print('max_q_vals shape', max_q_vals.shape)
    print('errors shape', errors.shape)
    print('buffer_counts shape', buffer_counts.shape)
    if 'Q_errors' in exp_data[0].keys():
        Q_errors = np.array(Q_errors)
        print('Q_errors shape', Q_errors.shape)


    if exp_args['env_args']['env_name'] == 'gridEnv1':

        # max_q_vals
        max_q_vals = np.mean(max_q_vals, axis=0)
        max_q_vals = np.reshape(max_q_vals, (8,-1))

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(max_q_vals, linewidth=0.5, cmap="coolwarm")#, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/max_q_vals.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/max_q_vals.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Greedy Q-vals policy
        q_vals_greedy_policy = np.argmax(q_vals_summed, axis=1) # [S]
        q_vals_greedy_policy = np.reshape(q_vals_greedy_policy, (8,-1))

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(q_vals_greedy_policy, linewidth=0.5, cmap="coolwarm") #, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        #plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        #plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/q_vals_greedy_policy.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/q_vals_greedy_policy.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # max_e_vals
        max_e_vals = np.mean(max_e_vals, axis=0)
        max_e_vals = np.reshape(max_e_vals, (8,-1))
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(max_e_vals, linewidth=0.5, cmap="coolwarm")#, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/max_e_vals.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/max_e_vals.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # max estimated_e_vals
        max_estimated_e_vals = np.mean(max_estimated_e_vals, axis=0)
        max_estimated_e_vals = np.reshape(max_estimated_e_vals, (8,-1))
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(max_estimated_e_vals, linewidth=0.5, cmap="coolwarm", cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/max_estimated_e_vals.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/max_estimated_e_vals.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # greedy e-vals policy
        e_vals_greedy_policy = np.argmax(e_vals_summed, axis=1) # [S]
        e_vals_greedy_policy = np.reshape(e_vals_greedy_policy, (8,-1))

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(e_vals_greedy_policy, linewidth=0.5, cmap="coolwarm") #, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        #plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        #plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/e_vals_greedy_policy.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/e_vals_greedy_policy.png', bbox_inches='tight', pad_inches=0)
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

        sns.heatmap(buffer_counts, annot=labels, linewidth=0.5, cmap="coolwarm", mask=mask_array, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        #plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        #plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/buffer_counts.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/buffer_counts.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

        # errors: abs(Q_phi - Q*)
        errors = np.mean(errors, axis=0)
        errors = np.reshape(errors, (8,-1))
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(errors, linewidth=0.5, cmap="coolwarm")#, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/errors.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/errors.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

        if 'Q_errors' in exp_data[0].keys():
            # errors: abs(Q_phi - (r+gamma*max(Q[s_t1,:])))

            Q_errors = np.mean(Q_errors, axis=0)
            Q_errors = np.reshape(Q_errors, (8,-1))
            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            sns.heatmap(Q_errors, linewidth=0.5, cmap="coolwarm") #, cbar=False)

            #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
            #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

            plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
            plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

            plt.xticks([]) # remove the tick marks by setting to an empty list
            plt.yticks([]) # remove the tick marks by setting to an empty list
            plt.axes().set_aspect('equal') #set the x and y axes to the same scale
            plt.grid()
            plt.savefig(f'{e_vals_folder_path}/Q_errors.png', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'{e_vals_folder_path}/Q_errors.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()

    elif exp_args['env_args']['env_name'] == 'gridEnv4':

        colormap = sns.color_palette("coolwarm", as_cmap=True)
        colormap.set_bad("black")

        # max_q_vals
        max_q_vals = np.mean(max_q_vals, axis=0)
        max_q_vals = np.reshape(max_q_vals, (8,-1))
        max_q_vals = np.delete(max_q_vals, 0, 0)
        max_q_vals[3,[1,2,3,4,5,6]] = np.nan
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(max_q_vals, linewidth=0.5, cmap=colormap)#, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/max_q_vals.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/max_q_vals.png', bbox_inches='tight', pad_inches=0)
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
        # plt.savefig(f'{e_vals_folder_path}/q_vals_greedy_policy.pdf', bbox_inches='tight', pad_inches=0)
        # plt.savefig(f'{e_vals_folder_path}/q_vals_greedy_policy.png', bbox_inches='tight', pad_inches=0)
        # plt.close()

        # max_e_vals
        max_e_vals = np.mean(max_e_vals, axis=0)
        max_e_vals = np.reshape(max_e_vals, (8,-1))
        max_e_vals = np.delete(max_e_vals, 0, 0)
        max_e_vals[3,[1,2,3,4,5,6]] = np.nan
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(max_e_vals, linewidth=0.5, cmap=colormap)#, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/max_e_vals.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/max_e_vals.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # greedy e-vals policy
        e_vals_greedy_policy = np.argmax(e_vals_summed, axis=1) # [S]
        e_vals_greedy_policy = np.reshape(e_vals_greedy_policy, (8,-1))
        e_vals_greedy_policy = np.delete(e_vals_greedy_policy, 0, 0)
        # e_vals_greedy_policy[3,[1,2,3,4,5,6]] = np.nan

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(e_vals_greedy_policy, linewidth=0.5, cmap="coolwarm") #, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        #plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
        #plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/e_vals_greedy_policy.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/e_vals_greedy_policy.png', bbox_inches='tight', pad_inches=0)
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
        plt.savefig(f'{e_vals_folder_path}/buffer_counts.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/buffer_counts.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

        # errors
        errors = np.mean(errors, axis=0)
        errors = np.reshape(errors, (8,-1))
        errors = np.delete(errors, 0, 0)
        errors[3,[1,2,3,4,5,6]] = np.nan
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        sns.heatmap(errors, linewidth=0.5, cmap="coolwarm")#, cbar=False)

        #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
        #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

        plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
        plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

        plt.xticks([]) # remove the tick marks by setting to an empty list
        plt.yticks([]) # remove the tick marks by setting to an empty list
        plt.axes().set_aspect('equal') #set the x and y axes to the same scale
        plt.grid()
        plt.savefig(f'{e_vals_folder_path}/errors.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{e_vals_folder_path}/errors.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

        if 'Q_errors' in exp_data[0].keys():
            # errors: abs(Q_phi - (r+gamma*max(Q[s_t1,:])))

            Q_errors = np.mean(Q_errors, axis=0)
            Q_errors = np.reshape(Q_errors, (8,-1))
            Q_errors = np.delete(Q_errors, 0, 0)
            Q_errors[3,[1,2,3,4,5,6]] = np.nan

            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            sns.heatmap(Q_errors, linewidth=0.5, cmap="coolwarm")#, cbar=False)

            #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
            #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

            plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
            plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

            plt.xticks([]) # remove the tick marks by setting to an empty list
            plt.yticks([]) # remove the tick marks by setting to an empty list
            plt.axes().set_aspect('equal') #set the x and y axes to the same scale
            plt.grid()
            plt.savefig(f'{e_vals_folder_path}/Q_errors.png', bbox_inches='tight', pad_inches=0)
            plt.savefig(f'{e_vals_folder_path}/Q_errors.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    main(exp_id=None, val_iter_exp=None)
