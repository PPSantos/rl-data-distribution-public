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


def main(exp_id):

    print('Arguments (analysis/plots_qlearning.py):')
    print('Exp. id: {0}'.format(exp_id))

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

    # Open data.
    print(f"Opening experiment {exp_id}")
    exp_path = DATA_FOLDER_PATH + exp_id
    with open(exp_path + "/train_data.json", 'r') as f:
        exp_data = json.load(f)
        exp_data = json.loads(exp_data)
    f.close()

    # Parse data.
    data = {}

    # Q_vals field.
    # data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,(S),S,A]

    # rollouts_rewards field.
    data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data]) # [R,(E)]

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    
    for (i, data) in enumerate(data['episode_rewards']):
        plt.plot(data)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.savefig('{0}/episode_rewards.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

