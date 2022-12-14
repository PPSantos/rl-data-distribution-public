import os
import numpy as np 
import pandas as pd
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('ggplot')

#################################################################
ENVS_LABELS = {
    'gridEnv1': 'Grid 1',
    'gridEnv2': 'Grid 2',
    'multiPathEnv': 'Multi-path',
    'pendulum': 'Pendulum',
    'mountaincar': 'Mountain car',
    'cartpole': 'Cartpole',
}

MINIMUM_REWARD = {
    'gridEnv1': 0.0,
    'gridEnv2': 0.0,
    'multiPathEnv': 0.0,
    'pendulum': -100.0,
    'mountaincar': -200.0,
    'cartpole': 0.0,
}
MAXIMUM_REWARD = {
    'gridEnv1': 36.0,
    'gridEnv2': 37.0,
    'multiPathEnv': 15.0,
    'pendulum': -8.5, 
    'mountaincar': -93.0,
    'cartpole': 200.0,
}

# Absolute path to folder containing experiments data.
CSV_PATH = '/home/pedrosantos/git/rl-data-distribution/data/parsed_results.csv'
DATA = pd.read_csv(CSV_PATH)

#################################################################
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

FIGURE_X = 7.0
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

    fig, axs = plt.subplots(ncols=3, nrows=2,
                            constrained_layout=True)
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    # fig.suptitle('plt.subplots()')

    axs_idxs = {'gridEnv1': [0,0],
                'gridEnv2': [0,1],
                'multiPathEnv': [0,2],
                'pendulum': [1,0],
                'mountaincar': [1,1],
                'cartpole': [1,2],
            }
    for env_id, env_lbl in ENVS_LABELS.items():

        """
            DQN.
        """
        filtered_df = DATA.loc[
                        (DATA['env_id']==env_id) &
                        (DATA['algo_id']=='offline_dqn') &
                        (DATA['dataset_type_id'].isin(['eps-greedy', 'boltzmann'])) &
                        (DATA['force_dataset_coverage']==True)
                    ]
        Y = filtered_df['rollouts_rewards_final'].to_numpy()
        Y_normalized = (Y - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id])
        filtered_df['normalized_rollouts_rewards_final'] = Y_normalized

        X = filtered_df['kl_dist']

        # Points.
        ax = axs[axs_idxs[env_id][0],axs_idxs[env_id][1]]
        p_1 = ax.scatter(X, Y_normalized, alpha=0.25)

        # Lines.
        bins = np.linspace(min(filtered_df['kl_dist']),
                    max(filtered_df['kl_dist']), 7)
        bins_group = pd.cut(filtered_df['kl_dist'], bins)
        grouped = filtered_df.groupby(bins_group)['normalized_rollouts_rewards_final'].agg(['mean', 'count'])
        grouped = grouped.to_numpy()
        Y = grouped[:,0]
        mask = np.isfinite(Y)
        Y = Y[mask]
        bins_to_plot = bins[:-1] + ((bins[1]-bins[0]) / 2)
        bins_to_plot = bins_to_plot[mask]
        p_1, = ax.plot(bins_to_plot, Y, zorder=1)

        """
            CQL.
        """
        filtered_df = DATA.loc[
                        (DATA['env_id']==env_id) &
                        (DATA['algo_id']=='offline_cql') &
                        (DATA['dataset_type_id'].isin(['eps-greedy', 'boltzmann'])) &
                        (DATA['force_dataset_coverage']==True)
                    ]
        Y = filtered_df['rollouts_rewards_final'].to_numpy()
        Y_normalized = (Y - MINIMUM_REWARD[env_id]) / (MAXIMUM_REWARD[env_id] - MINIMUM_REWARD[env_id])
        filtered_df['normalized_rollouts_rewards_final'] = Y_normalized

        X = filtered_df['kl_dist']

        # Points.
        ax = axs[axs_idxs[env_id][0],axs_idxs[env_id][1]]
        p_2 = ax.scatter(X, Y_normalized, alpha=0.25)

        # Lines.
        bins = np.linspace(min(filtered_df['kl_dist']),
                    max(filtered_df['kl_dist']), 7)
        bins_group = pd.cut(filtered_df['kl_dist'], bins)
        grouped = filtered_df.groupby(bins_group)['normalized_rollouts_rewards_final'].agg(['mean', 'count'])
        grouped = grouped.to_numpy()
        Y = grouped[:,0]
        mask = np.isfinite(Y)
        Y = Y[mask]
        bins_to_plot = bins[:-1] + ((bins[1]-bins[0]) / 2)
        bins_to_plot = bins_to_plot[mask]
        p_2, = ax.plot(bins_to_plot, Y, zorder=1)

        if env_id in ['gridEnv1', 'pendulum']:
            ax.set_ylabel('Norm. reward')
        if env_id in ['mountaincar', 'pendulum', 'cartpole']:
            ax.set_xlabel(r'$\min_{\pi^*}\mathcal{D}_f(d_{\pi^*}||\mu$)')

        ax.set_title(env_lbl)

    fig.legend((p_1, p_2), ('DQN', 'CQL'), bbox_to_anchor=(0.5, 0., 0.2, 0.025), ncol=2)

    plt.savefig(f'{output_folder}/distance_plot.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/distance_plot.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
