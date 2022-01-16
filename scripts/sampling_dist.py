import os
from tqdm import tqdm
import json
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
sns.set_style(style='white')
plt.style.use('ggplot')
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
#matplotlib.rcParams.update({'font.size': 13})

FIGURE_X = 8.0
FIGURE_Y = 6.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'

def create_exp_name(env_name):
    return env_name + \
        '_' + 'sampling_dist' + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


def main(env_name, policy, num_episodes=10_000):

    # Setup experiment data folder.
    exp_name = create_exp_name(env_name)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nSampling dist. ID:', exp_name)
    print('utils/sampling_dist.py arguments:')
    print('env_name', env_name, ', num_episodes', num_episodes)

    # Setup plots folder.
    plots_folder_path = PLOTS_FOLDER_PATH + exp_name
    os.makedirs(plots_folder_path, exist_ok=True)

    # Load train (and rollouts) environment.
    env, env_grid_spec, rollouts_envs = env_suite.get_env(env_name)

    # Rollout policy.
    episode_rewards = []
    sa_counts = np.zeros((env.num_states, env.num_actions))

    for _ in tqdm(range(num_episodes)):

        obs = env.reset()
        s_t = env.get_state()

        done = False
        episode_cumulative_reward = 0
        while not done:

            # Pick action.
            a_t = policy(s_t)

            # Env step.
            obs_t1, r_t1, done, info = env.step(a_t)
            s_t1 = env.get_state()

            # Log data.
            episode_cumulative_reward += r_t1
            sa_counts[s_t,a_t] += 1

            s_t = s_t1

        episode_rewards.append(episode_cumulative_reward)

    data = {}
    data['episode_rewards'] = episode_rewards
    data['sa_counts'] = sa_counts # [S,A]

    sampling_dist = sa_counts / np.sum(sa_counts) # [S,A]
    sampling_dist_flattened = sampling_dist.flatten() # [S*A]
    #print(sampling_dist_flattened)
    print('(S,A) dist. entropy:', scipy.stats.entropy(sampling_dist_flattened))
    data['sampling_dist'] = sampling_dist_flattened

    # Optionally plot sampling distribution.
    if env_name in ('gridEnv1', 'gridEnv4'):

        # 2D plot.
        s_counts = np.sum(sa_counts, axis=1) # [S]
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

    return exp_path + "/data.json", exp_name
