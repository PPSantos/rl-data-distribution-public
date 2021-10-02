import os
from tqdm import tqdm
import json
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


def _build_policy_func(num_switched_actions=20, epsilon=0.3):
    print('num_switched_actions=', num_switched_actions)
    print('epsilon=', epsilon)

    # gridEnv 4 top optimal trajectory.
    actions = [0,0,0,0,0,0,0,0,
               4,4,4,4,4,4,4,2,
               4,4,4,4,4,4,4,2,
               4,4,4,4,4,4,4,2,
               1,0,0,0,0,0,0,4,
               4,4,4,4,4,4,4,1,
               4,4,4,4,4,4,4,1,
               4,4,4,4,4,4,4,1]
    print(actions)

    switched_actions_idxs = np.random.choice(np.arange(len(actions)),
                            size=num_switched_actions,replace=False)
    for idx in switched_actions_idxs:
        new_action = np.random.randint(low=0,high=5)
        actions[idx] = new_action

    print(actions)
    
    def p(s):
        if np.random.rand() <= epsilon:
            return np.random.randint(low=0,high=5)
        else:
            return actions[s]

    return p


DEFAULT_SAMPLING_DIST_ARGS = {
    # WARNING: only works with tabular/grid envs.

    'num_episodes': 10_000,

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

def main(args=None, policy=None):

    if not args:
        args = DEFAULT_SAMPLING_DIST_ARGS
    
    if not policy:
        policy = _build_policy_func()

    # Setup experiment data folder.
    exp_name = create_exp_name(args)
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nSampling dist. ID:', exp_name)
    print('utils/sampling_dist.py arguments:')
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

    # Rollout policy.
    episode_rewards = []
    sa_counts = np.zeros((env.num_states, env.num_actions))

    for _ in tqdm(range(args['num_episodes'])):

        s_t = env.reset()

        done = False
        episode_cumulative_reward = 0
        while not done:

            # Pick action.
            a_t = policy(s_t)

            # Env step.
            s_t1, r_t1, done, info = env.step(a_t)

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

    if args['env_args']['env_name'] in ('gridEnv1', 'gridEnv4'):

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

if __name__ == "__main__":
    main()
