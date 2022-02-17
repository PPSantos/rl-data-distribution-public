import os
import re
import json
import tarfile
import numpy as np 
import pandas as pd
import scipy
from scipy.spatial import distance
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})


#################################################################
ENVS_LABELS = {
    'gridEnv1': 'Grid 1',
    'gridEnv2': 'Grid 2',
    'multiPathEnv': 'Multi-path',
    'pendulum': 'Pendulum',
    'mountaincar': 'Mountain car',
}

MINIMUM_REWARD = {
    'gridEnv1': 0.0,
    'gridEnv2': 0.0,
    'multiPathEnv': 0.0,
    'pendulum': -100.0,
    'mountaincar': -200.0,
}
MAXIMUM_REWARD = {
    'gridEnv1': 36.0,
    'gridEnv2': 37.0,
    'multiPathEnv': 15.0,
    'pendulum': -8.5, 
    'mountaincar': -93.0,
}

# Absolute path to folder containing experiments data.
CSV_PATH = '/home/pedrosantos/git/rl-data-distribution/data/parsed_data.csv'
DATA = pd.read_csv(CSV_PATH)
#################################################################

FIGURE_X = 7.0
FIGURE_Y = 4.0

PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

def get_epsilon(info_text):
    p = 'epsilon=[\d]*[.][\d]+'
    result = re.search(p, info_text).group(0)
    epsilon = float(result.split('=')[1])
    return epsilon

def mean_agg_func(samples: np.ndarray, num_resamples: int=25_000):
    """
        Computes mean.
    """
    # Point estimation.
    point_estimate = np.mean(samples)
    # Confidence interval estimation.
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = np.mean(resampled, axis=0)
    confidence_interval = [np.percentile(point_estimations, 5),
                           np.percentile(point_estimations, 95)]
    return point_estimate, confidence_interval

def main():

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + '/article_plots'
    os.makedirs(output_folder, exist_ok=True)

    fig, axs = plt.subplots(ncols=3, nrows=2,
                            constrained_layout=True)
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    axs_idxs = {'gridEnv1': [0,0],
                'gridEnv2': [0,1],
                'multiPathEnv': [0,2],
                'pendulum': [1,0],
                'mountaincar': [1,1],
                'cartpole': [1,2]}
    for env_id, env_lbl in ENVS_LABELS.items():

        """
            DQN.
        """
        filtered_df = DATA.loc[
                        (DATA['env_id']==env_id) &
                        (DATA['algo_id']=='offline_dqn') &
                        (DATA['dataset_type_id'].isin(['eps-greedy'])) &
                        (DATA['force_dataset_coverage']==True)
                    ]

        Y = filtered_df['rollouts_rewards_final'].to_numpy()
        Y_normalized = (Y - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id])
        filtered_df['normalized_rollouts_rewards_final'] = Y_normalized
        filtered_df['epsilon'] = filtered_df.apply(lambda row : get_epsilon(row['info_text']), axis=1)

        # X-axis: epsilon values.
        X = filtered_df['epsilon'].to_numpy()

        # Y-axis: normalized mean reward + std.
        stds = []
        means = []
        for exp_id in filtered_df['id'].to_numpy():

            # Load train data.
            print('Opening exp_id', exp_id)
            exp_folder_path = DATA_FOLDER_PATH + exp_id + '.tar.gz'
            tar = tarfile.open(exp_folder_path)
            data_file = tar.extractfile("{0}/train_data.json".format(exp_id))
            exp_data = json.load(data_file)
            exp_data = json.loads(exp_data)

            data = {}
            data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(Steps)]

            stds.append(np.std(data['rollouts_rewards'][:,-1]))
            means.append(np.mean(data['rollouts_rewards'][:,-1]))

        Y = (np.array(means) - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id]) 
        Y_std = (np.array(stds) - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id]) 

        # Plot.
        ax = axs[axs_idxs[env_id][0],axs_idxs[env_id][1]]
        p_1, = ax.plot(X, Y, alpha=0.8)

        # ax.fill_between(X, min(Y), max(Y),
        #                 color=p_1[0].get_color(), alpha=0.15)

        """
            CQL.
        """
        filtered_df = DATA.loc[
                        (DATA['env_id']==env_id) &
                        (DATA['algo_id']=='offline_cql') &
                        (DATA['dataset_type_id'].isin(['eps-greedy'])) &
                        (DATA['force_dataset_coverage']==True)
                    ]

        Y = filtered_df['rollouts_rewards_final'].to_numpy()
        Y_normalized = (Y - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id])
        filtered_df['normalized_rollouts_rewards_final'] = Y_normalized

        filtered_df['epsilon'] = filtered_df.apply(lambda row : get_epsilon(row['info_text']), axis = 1)

        X = filtered_df['epsilon'].to_numpy()
        Y = filtered_df['normalized_rollouts_rewards_final'].to_numpy()

        # Plot.
        ax = axs[axs_idxs[env_id][0],axs_idxs[env_id][1]]
        p_2, = ax.plot(X, Y, alpha=0.8)

        ax.set_ylabel('Norm. reward')
        ax.set_xlabel(r'$\epsilon$')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        ax.set_title(env_lbl)

    for ax in axs.flat:
        ax.label_outer()

    fig.legend((p_1, p_2), ('DQN', 'CQL'), bbox_to_anchor=(0.5, 0., 0.2, 0.025), ncol=2)

    plt.savefig(f'{output_folder}/coverage_plot.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/coverage_plot.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
