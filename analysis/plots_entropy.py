from cProfile import label
import os
import json
import tarfile
from tokenize import group
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

from statsmodels.nonparametric.smoothers_lowess import lowess

#################################################################
ENVS_LABELS = {
    'gridEnv1': 'Grid 1',
    'gridEnv2': 'Grid 2',
    'multiPathEnv': 'Multi-path',
    'pendulum': 'Pendulum',
    'mountaincar': 'Mountain car',
}

MAX_ENTROPY = {
    'gridEnv1': 5.8,
    'gridEnv2': 5.6,
    'multiPathEnv': 4.9,
    'pendulum': 9.4,
    'mountaincar': 8.9,
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
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

FIGURE_X = 6.0
FIGURE_Y = 4.0

PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

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

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for env_id, env_lbl in ENVS_LABELS.items():

        filtered_df = DATA.loc[
                        (DATA['env_id']==env_id) &
                        (DATA['algo_id']=='offline_dqn') &
                        (DATA['dataset_type_id'].isin(['dirichlet'])) # &
                        # (DATA['force_dataset_coverage']==force_coverage_type)
                    ]

        print('Env', env_id)
        print('max reward', filtered_df['rollouts_rewards_final'].max())
        print('min reward', filtered_df['rollouts_rewards_final'].min())
        print('max entropy', filtered_df['dataset_entropy'].max())
        print('min entropy', filtered_df['dataset_entropy'].min())

        Y = filtered_df['rollouts_rewards_final'].to_numpy()
        Y_normalized = (Y - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id])

        X = filtered_df['dataset_entropy'].to_numpy()
        X_normalized = X / MAX_ENTROPY[env_id]

        filtered_df['normalized_dataset_entropy'] = X_normalized
        filtered_df['normalized_rollouts_rewards_final'] = Y_normalized

        bins = np.linspace(min(filtered_df['normalized_dataset_entropy']),
                    max(filtered_df['normalized_dataset_entropy']), 7)
        print('bins', bins)
        # bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bins_group = pd.cut(filtered_df['normalized_dataset_entropy'], bins)
        grouped = filtered_df.groupby(bins_group)['normalized_rollouts_rewards_final'].agg(['mean', 'count', 'std'])
        print(grouped)

        plt.scatter(X_normalized, Y_normalized, alpha=0.8, label=env_lbl)

        # Y = grouped.to_numpy()
        # bins_to_plot = bins[:-1] + ((bins[1]-bins[0]) / 2)
        # print('bins_to_plot', bins_to_plot)
        # plt.plot(bins_to_plot, Y[:,0], zorder=1) # label=env_lbl,

        # sm_x, sm_y = lowess(Y_normalized, X_normalized, frac=0.7, 
        #                         it=3, return_sorted = True).T
        # plt.plot(sm_x, sm_y, label=env_lbl) #, label='LOWESS')

        plt.ylabel('Normalized reward')
        plt.xlabel(r'$\mathcal{H}(\mu) / \mathcal{H}(\mathcal{U})$')

    plt.legend(loc=6)
    plt.savefig(f'{output_folder}/entropy_plot.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'{output_folder}/entropy_plot.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
