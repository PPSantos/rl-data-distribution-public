import sys
import os
from tqdm import tqdm
import json
import time
import numpy as np
import pathlib
from datetime import datetime
import scipy

from utils.json_utils import NumpyEncoder
from envs import env_suite

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})
sns.set_style(style='white')

FIGURE_X = 8.0
FIGURE_Y = 6.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'


ARGS = {

    'dirichlet_alpha_coef': 5.0,

    # Env. arguments.
    'env_args': {
        'env_name': 'gridEnv4',
        'dim_obs': 8,
        'time_limit': 25,
        'tabular': True, # do not change.
        'smooth_obs': False,
        'one_hot_obs': False,
    },

}

def create_exp_name(args):
    return args['env_args']['env_name'] + \
        '_' + 'sampling_dist' + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def main():

    args = ARGS

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('train.py arguments:')
    print(args)

    # Setup plots folder.
    plots_folder_path = PLOTS_FOLDER_PATH + exp_name
    os.makedirs(plots_folder_path, exist_ok=True)

    # Load train (and rollouts) environment.
    env_name = args['env_args']['env_name']
    if env_name in env_suite.CUSTOM_GRID_ENVS.keys():
        env, env_grid_spec, rollouts_envs = env_suite.get_custom_grid_env(**args['env_args'],
                                                        absorb=False)
    else:
        raise ValueError('Only implemented for grid/tabular environments.')
        # env, rollouts_envs = env_suite.get_env(env_name, seed=time_delay)
        # env_grid_spec = None

    # Create sampling dist.
    sampling_dist_size = env.num_states * env.num_actions
    sampling_dist = np.random.dirichlet([args['dirichlet_alpha_coef']]*sampling_dist_size)
    print('(S,A) dist. entropy:', scipy.stats.entropy(sampling_dist))

    data = {}
    data['sampling_dist'] = sampling_dist

    # 2D plot.
    sampling_dist = sampling_dist.reshape(env.num_states, env.num_actions)
    s_counts = np.sum(sampling_dist, axis=1)
    s_counts = np.reshape(s_counts, (8,-1))
    #s_counts[0,7] = np.nan
    #s_counts[7,0] = np.nan
    #mask_array = np.ma.masked_invalid(s_counts).mask
    labels = s_counts / np.nansum(s_counts) * 100
    labels = np.around(labels, decimals=1)
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    sns.heatmap(s_counts, annot=labels, linewidth=0.5, cmap="coolwarm", cbar=False)
    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig(f'{plots_folder_path}/s_counts.png', bbox_inches='tight', pad_inches=0)
    #plt.savefig(f'{plots_folder_path}/s_counts.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # 3D plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(8)
    y = np.arange(8)
    _xx, _yy = np.meshgrid(x, y)
    x, y = _xx.ravel(), _yy.ravel()
    labels = labels.T
    top = labels.flatten()
    bottom = np.zeros_like(top)
    width = depth = 1
    #light = matplotlib.colors.LightSource(azdeg=200., altdeg=45)
    ax.bar3d(x, y, bottom, width, depth, top, shade=True)#, lightsource=light)
    ax.view_init(elev=20., azim=30)
    ax.set_zlim(0,20)
    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.savefig(f'{plots_folder_path}/s_counts_3d.png', bbox_inches='tight', pad_inches=0)
    #plt.savefig(f'{plots_folder_path}/s_counts_3d.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Store data.
    f = open(exp_path + "/data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()


if __name__ == "__main__":
    main()
