import sys
import os
from tqdm import tqdm
import json
import time
import numpy as np
import pathlib
from datetime import datetime
import scipy

from rlutil.json_utils import NumpyEncoder
from envs import env_suite

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

FIGURE_X = 8.0
FIGURE_Y = 6.0


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'

sns.set_style(style='white')

# gridEnv1
""" def policy(s):
    if (s // 8) == 0:
        return 4 # RIGHT
    else:
        return 1 # UP """

# gridEnv2 (optimal top trajectory)
def policy(s):
    if s in (0,1,2,3,4,5,6,
             8,9,10,11,12,13,14,
             16,17,18,19,20,21,22,
             32,33,34,35,36,37,38
             40,41,42,43,44,45,46
             48,49,50,51,52,53,54):
        return 4 # RIGHT
    elif s in (7,15,23):
        return 2 # DOWN
    elif s in(39,47,55):
        return 1 # UP
    elif s in (24,) # INITIAL STATE.
        return 1 # UP
    elif s in (31,)
        return # 4 RIGHT
    else:
        raise ValueError("policy error")

""" # gridEnv2 (optimal bottom trajectory)
def policy(s):
    if s in (0,1,2,3,4,5,6,
             8,9,10,11,12,13,14,
             16,17,18,19,20,21,22,
             32,33,34,35,36,37,38
             40,41,42,43,44,45,46
             48,49,50,51,52,53,54):
        return 4 # RIGHT
    elif s in (7,15,23):
        return 2 # DOWN
    elif s in(39,47,55):
        return 1 # UP
    elif s in (24,) # INITIAL STATE.
        return 2 # DOWN
    elif s in (31,)
        return # 4 RIGHT
    else:
        raise ValueError("policy error") """

DEFAULT_TRAIN_ARGS = {
    # WARNING: only works with tabular/grid envs.

    'num_episodes': 10_000,
    'epsilon': 0.3,

    # Env. arguments.
    'env_args': {
        'env_name': 'gridEnv4',
        'dim_obs': 8,
        'time_limit': 50,
        'tabular': True, # do not change.
        'smooth_obs': False,
        'one_hot_obs': False,
    },

}

def create_exp_name(args):
    return args['env_args']['env_name'] + \
        '_' + 'sampling_dist' + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def rollout():

    args = DEFAULT_TRAIN_ARGS

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

    # Test policy.
    # for s in range(env.num_states):
    #     print('s', s, policy(s))

    # Rollout policy.
    episode_rewards = []
    sa_counts = np.zeros((env.num_states,env.num_actions))

    for episode in tqdm(range(args['num_episodes'])):

        s_t = env.reset()
        # print('ENV RESET.')

        done = False
        episode_cumulative_reward = 0
        while not done:

            # Pick action (epsilon-greedy policy).
            if np.random.rand() <= args['epsilon']:
                a_t = np.random.choice(env.num_actions)
            else:
                a_t = policy(s_t)

            # Env step.
            s_t1, r_t1, done, info = env.step(a_t)

            # print('s_t', s_t)
            # print('a_t', a_t)
            # print('r_t1', r_t1)

            # Log data.
            episode_cumulative_reward += r_t1
            sa_counts[s_t,a_t] += 1

            # env.render()

            s_t = s_t1

        episode_rewards.append(episode_cumulative_reward)

    data = {}
    data['episode_rewards'] = episode_rewards
    data['sa_counts'] = sa_counts # [S,A]

    sa_dist = sa_counts / np.sum(sa_counts) # [S,A]
    sa_dist_flattened = sa_dist.flatten() # [S]
    print(sa_dist_flattened)
    print('(S,A) dist. entropy:', scipy.stats.entropy(sa_dist_flattened))
    data['sampling_dist'] = sa_dist_flattened

    # 2D plot.
    s_counts = np.sum(sa_counts, axis=1)
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
    rollout()
