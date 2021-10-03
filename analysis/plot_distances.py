import os
import json
import numpy as np
import pathlib
from datetime import datetime
import scipy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

FIGURE_X = 8.0
FIGURE_Y = 6.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'


OPTIMAL_DISTS_IDS = [
'gridEnv4_sampling_dist_2021-10-02-15-59-29', # GridEnv4 top trajectory.
'gridEnv4_sampling_dist_2021-10-02-15-46-54', # GridEnv4 bottom trajectory.
]

SAMPLING_DISTS_IDS = [
'gridEnv4_sampling_dist_2021-10-01-15-32-07',
'gridEnv4_sampling_dist_2021-10-01-15-33-08',
'gridEnv4_sampling_dist_2021-10-01-15-33-28',
'gridEnv4_sampling_dist_2021-10-01-15-33-43',
'gridEnv4_sampling_dist_2021-10-01-15-34-02',

'gridEnv4_sampling_dist_2021-10-01-21-14-42',
'gridEnv4_sampling_dist_2021-10-01-21-15-23',
'gridEnv4_sampling_dist_2021-10-01-21-15-59',
'gridEnv4_sampling_dist_2021-10-01-21-16-34',
'gridEnv4_sampling_dist_2021-10-01-21-16-57',
'gridEnv4_sampling_dist_2021-10-01-21-17-17',
]

OFFLINE_DQN_EXP_IDS = [
'gridEnv4_offline_dqn_2021-10-01-15-44-34',
'gridEnv4_offline_dqn_2021-10-01-16-28-41',
'gridEnv4_offline_dqn_2021-10-01-17-13-19',
'gridEnv4_offline_dqn_2021-10-01-17-57-41',
'gridEnv4_offline_dqn_2021-10-01-18-41-59',

'gridEnv4_offline_dqn_2021-10-01-21-19-38',
'gridEnv4_offline_dqn_2021-10-01-22-44-53',
'gridEnv4_offline_dqn_2021-10-02-00-11-13',
'gridEnv4_offline_dqn_2021-10-02-01-37-16',
'gridEnv4_offline_dqn_2021-10-02-03-03-09',
'gridEnv4_offline_dqn_2021-10-02-04-29-15'
]

METRIC_TO_PLOT = 'qvals_avg_error'


def main():

    output_folder = PLOTS_FOLDER_PATH + '/distances_plots'
    os.makedirs(output_folder, exist_ok=True)

    optimal_dists = []
    for optimal_dist_id in OPTIMAL_DISTS_IDS:
        optimal_dist_path = DATA_FOLDER_PATH + optimal_dist_id + '/data.json'
        print(optimal_dist_path)
        with open(optimal_dist_path, 'r') as f:
            data = json.load(f)
            d = np.array(json.loads(data)['sampling_dist'])
        f.close()
        optimal_dists.append(d)
    #print('optimal_dists', optimal_dists)

    sampling_dists = []
    for sampling_dist_id in SAMPLING_DISTS_IDS:
        sampling_dist_path = DATA_FOLDER_PATH + sampling_dist_id + '/data.json'
        with open(sampling_dist_path, 'r') as f:
            data = json.load(f)
            d = np.array(json.loads(data)['sampling_dist'])
        f.close()
        sampling_dists.append(d)
    #print('sampling_dists', sampling_dists)

    offline_dqn_metrics = []
    for offline_dqn_exp_id in OFFLINE_DQN_EXP_IDS:
        exp_metrics_path = PLOTS_FOLDER_PATH + offline_dqn_exp_id + '/scalar_metrics.json'
        with open(exp_metrics_path, 'r') as f:
            d = json.load(f)
        f.close()
        offline_dqn_metrics.append(d)
    print('offline_dqn_metrics', offline_dqn_metrics)

    wass_dists = []
    ratio_dists = []
    kl_dists = []
    for sampling_dist in sampling_dists:

        wass_dist = np.min([scipy.stats.wasserstein_distance(optimal_dist, sampling_dist)
                        for optimal_dist in optimal_dists])
        ratio_dist = np.min([np.max(optimal_dist/(sampling_dist+1e-06))
                        for optimal_dist in optimal_dists])
        kl_dist = np.min([scipy.stats.entropy(optimal_dist,sampling_dist)
                        for optimal_dist in optimal_dists])

        wass_dists.append(wass_dist)
        ratio_dists.append(ratio_dist)
        kl_dists.append(kl_dist)

    print('wass_dists', wass_dists)
    print('ratio_dists', ratio_dists)
    print('kl_dists', kl_dists)

    data_to_plot = [x[METRIC_TO_PLOT] for x in offline_dqn_metrics]
    print(data_to_plot)

    # KL-div. plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    Y = [y for _, y in sorted(zip(kl_dists, data_to_plot))]
    plt.plot(sorted(kl_dists), Y)
    plt.scatter(sorted(kl_dists), Y)
    plt.xlabel('KL div.')
    plt.ylabel(METRIC_TO_PLOT)
    plt.savefig(f'{output_folder}/{METRIC_TO_PLOT}_kl_div.png'.format(), bbox_inches='tight', pad_inches=0)

    # Ratio dist. plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    Y = [y for _, y in sorted(zip(ratio_dists, data_to_plot))]
    plt.plot(sorted(ratio_dists), Y)
    plt.scatter(sorted(ratio_dists), Y)
    plt.xlabel('Ratio dist.')
    plt.ylabel(METRIC_TO_PLOT)
    plt.savefig(f'{output_folder}/{METRIC_TO_PLOT}_ratio_dist.png'.format(), bbox_inches='tight', pad_inches=0)

    # Wass. dist. plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    Y = [y for _, y in sorted(zip(wass_dists, data_to_plot))]
    plt.plot(sorted(wass_dists), Y)
    plt.scatter(sorted(wass_dists), Y)
    plt.xlabel('Wass. dist.')
    plt.ylabel(METRIC_TO_PLOT)
    plt.savefig(f'{output_folder}/{METRIC_TO_PLOT}_wass_dist.png'.format(), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
