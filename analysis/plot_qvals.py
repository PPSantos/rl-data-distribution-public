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
        val_iter_exp = c_args.val_iter_exp
        
    print('Arguments (analysis/plot_state_qval.py):')
    print('Exp. id: {0}'.format(exp_id))
    print('Val. iter. exp. id: {0}\n'.format(val_iter_exp))

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/Q-values/'
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

    is_grid_env = exp_args['env_name'] in env_suite.CUSTOM_GRID_ENVS.keys()
    print('is_grid_env:', is_grid_env)

    if is_grid_env:
        lateral_size = int(np.sqrt(len(val_iter_data['policy']))) # Assumes env. is a square.

    """
        Q-values plots.
    """
    if is_grid_env:

        while True:

            print("\nEnter state to plot Q-val")
            line = int(input("\tline: "))
            col = int(input("\tcolumn: "))
            state_to_plot = lateral_size*line+col
            print(f'line={line}, col={col}: state={state_to_plot}')

            # Display val. iter. Q-values.
            print('Value iteration Q-values:')
            for action in range(val_iter_data['Q_vals'].shape[-1]):
                print(f"Q({state_to_plot},{action})={val_iter_data['Q_vals'][state_to_plot,action]}")

            # Plot.
            num_cols = 3
            y_axis_range = [0,11]
            num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [E,S,A]

                for action in range(run_Q_vals.shape[-1]):
                    
                    # Plot predicted Q-values.
                    Y = run_Q_vals[:,state_to_plot,action]
                    X = np.linspace(1, len(Y), len(Y))
                    l = ax.plot(X, Y, label=f'Action {action}')

                    # Plot true Q-values.
                    ax.hlines(val_iter_data['Q_vals'][state_to_plot,action],
                            xmin=0, xmax=run_Q_vals.shape[0], linestyles='--', color=l[0].get_color())

                ax.legend()
                ax.set_ylim(y_axis_range)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Q-value')
                ax.set_title(label=f"Run {i}")

                i += 1

            fig.suptitle(f'Q-values, state {state_to_plot}; line {line}, col {col}')

            #plt.savefig('{0}/Q_values_s{1}-{2}-{3}.pdf'.format(output_folder, state_to_plot, line, col), bbox_inches='tight', pad_inches=0)
            plt.savefig('{0}/Q_values_s{1}-{2}-{3}.png'.format(output_folder, state_to_plot, line, col), bbox_inches='tight', pad_inches=0)

            plt.close()

if __name__ == "__main__":
    main(exp_id=None,val_iter_exp=None)
