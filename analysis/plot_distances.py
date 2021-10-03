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
'gridEnv4_sampling_dist_2021-10-03-09-56-57', 'gridEnv4_sampling_dist_2021-10-03-10-08-38', 'gridEnv4_sampling_dist_2021-10-03-10-20-18', 'gridEnv4_sampling_dist_2021-10-03-10-32-06', 'gridEnv4_sampling_dist_2021-10-03-10-43-45', 'gridEnv4_sampling_dist_2021-10-03-10-55-19', 'gridEnv4_sampling_dist_2021-10-03-11-06-51', 'gridEnv4_sampling_dist_2021-10-03-11-18-30', 'gridEnv4_sampling_dist_2021-10-03-11-30-06', 'gridEnv4_sampling_dist_2021-10-03-11-42-00', 'gridEnv4_sampling_dist_2021-10-03-11-53-33', 'gridEnv4_sampling_dist_2021-10-03-12-05-11', 'gridEnv4_sampling_dist_2021-10-03-12-17-00', 'gridEnv4_sampling_dist_2021-10-03-12-28-40', 'gridEnv4_sampling_dist_2021-10-03-12-40-11', 'gridEnv4_sampling_dist_2021-10-03-12-51-44', 'gridEnv4_sampling_dist_2021-10-03-13-03-29', 'gridEnv4_sampling_dist_2021-10-03-13-15-22', 'gridEnv4_sampling_dist_2021-10-03-13-26-57', 'gridEnv4_sampling_dist_2021-10-03-13-38-36', 'gridEnv4_sampling_dist_2021-10-03-13-50-01', 'gridEnv4_sampling_dist_2021-10-03-14-01-47', 'gridEnv4_sampling_dist_2021-10-03-14-14-25', 'gridEnv4_sampling_dist_2021-10-03-14-27-24', 'gridEnv4_sampling_dist_2021-10-03-14-40-29', 'gridEnv4_sampling_dist_2021-10-03-14-53-19', 'gridEnv4_sampling_dist_2021-10-03-15-05-49', 'gridEnv4_sampling_dist_2021-10-03-15-17-29', 'gridEnv4_sampling_dist_2021-10-03-15-28-51', 'gridEnv4_sampling_dist_2021-10-03-15-40-19', 'gridEnv4_sampling_dist_2021-10-03-15-51-28', 'gridEnv4_sampling_dist_2021-10-03-16-02-31', 'gridEnv4_sampling_dist_2021-10-03-16-17-53', 'gridEnv4_sampling_dist_2021-10-03-16-29-16', 'gridEnv4_sampling_dist_2021-10-03-16-41-03', 'gridEnv4_sampling_dist_2021-10-03-16-52-33', 'gridEnv4_sampling_dist_2021-10-03-17-04-11', 'gridEnv4_sampling_dist_2021-10-03-17-15-50', 'gridEnv4_sampling_dist_2021-10-03-17-27-25', 'gridEnv4_sampling_dist_2021-10-03-17-39-10', 'gridEnv4_sampling_dist_2021-10-03-17-50-56', 'gridEnv4_sampling_dist_2021-10-03-18-02-18', 'gridEnv4_sampling_dist_2021-10-03-18-13-36', 'gridEnv4_sampling_dist_2021-10-03-18-25-10', 'gridEnv4_sampling_dist_2021-10-03-18-36-40', 'gridEnv4_sampling_dist_2021-10-03-18-52-11', 'gridEnv4_sampling_dist_2021-10-03-19-04-02', 'gridEnv4_sampling_dist_2021-10-03-19-16-00', 'gridEnv4_sampling_dist_2021-10-03-19-27-55', 'gridEnv4_sampling_dist_2021-10-03-19-42-54',
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
'gridEnv4_offline_dqn_2021-10-02-04-29-15',
'gridEnv4_offline_dqn_2021-10-03-09-56-59', 'gridEnv4_offline_dqn_2021-10-03-10-08-39', 'gridEnv4_offline_dqn_2021-10-03-10-20-20', 'gridEnv4_offline_dqn_2021-10-03-10-32-07', 'gridEnv4_offline_dqn_2021-10-03-10-43-46', 'gridEnv4_offline_dqn_2021-10-03-10-55-20', 'gridEnv4_offline_dqn_2021-10-03-11-06-53', 'gridEnv4_offline_dqn_2021-10-03-11-18-32', 'gridEnv4_offline_dqn_2021-10-03-11-30-07', 'gridEnv4_offline_dqn_2021-10-03-11-42-01', 'gridEnv4_offline_dqn_2021-10-03-11-53-35', 'gridEnv4_offline_dqn_2021-10-03-12-05-13', 'gridEnv4_offline_dqn_2021-10-03-12-17-01', 'gridEnv4_offline_dqn_2021-10-03-12-28-42', 'gridEnv4_offline_dqn_2021-10-03-12-40-13', 'gridEnv4_offline_dqn_2021-10-03-12-51-45', 'gridEnv4_offline_dqn_2021-10-03-13-03-31', 'gridEnv4_offline_dqn_2021-10-03-13-15-24', 'gridEnv4_offline_dqn_2021-10-03-13-26-58', 'gridEnv4_offline_dqn_2021-10-03-13-38-38', 'gridEnv4_offline_dqn_2021-10-03-13-50-03', 'gridEnv4_offline_dqn_2021-10-03-14-01-49', 'gridEnv4_offline_dqn_2021-10-03-14-14-28', 'gridEnv4_offline_dqn_2021-10-03-14-27-26', 'gridEnv4_offline_dqn_2021-10-03-14-40-32', 'gridEnv4_offline_dqn_2021-10-03-14-53-22', 'gridEnv4_offline_dqn_2021-10-03-15-05-50', 'gridEnv4_offline_dqn_2021-10-03-15-17-31', 'gridEnv4_offline_dqn_2021-10-03-15-28-52', 'gridEnv4_offline_dqn_2021-10-03-15-40-21', 'gridEnv4_offline_dqn_2021-10-03-15-51-30', 'gridEnv4_offline_dqn_2021-10-03-16-02-32', 'gridEnv4_offline_dqn_2021-10-03-16-17-54', 'gridEnv4_offline_dqn_2021-10-03-16-29-17', 'gridEnv4_offline_dqn_2021-10-03-16-41-05', 'gridEnv4_offline_dqn_2021-10-03-16-52-35', 'gridEnv4_offline_dqn_2021-10-03-17-04-12', 'gridEnv4_offline_dqn_2021-10-03-17-15-52', 'gridEnv4_offline_dqn_2021-10-03-17-27-27', 'gridEnv4_offline_dqn_2021-10-03-17-39-11', 'gridEnv4_offline_dqn_2021-10-03-17-50-58', 'gridEnv4_offline_dqn_2021-10-03-18-02-19', 'gridEnv4_offline_dqn_2021-10-03-18-13-37', 'gridEnv4_offline_dqn_2021-10-03-18-25-12', 'gridEnv4_offline_dqn_2021-10-03-18-36-42', 'gridEnv4_offline_dqn_2021-10-03-18-52-13', 'gridEnv4_offline_dqn_2021-10-03-19-04-03', 'gridEnv4_offline_dqn_2021-10-03-19-16-02', 'gridEnv4_offline_dqn_2021-10-03-19-27-58', 'gridEnv4_offline_dqn_2021-10-03-19-42-56'
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
    plt.yscale('log')
    plt.ylabel(METRIC_TO_PLOT)
    plt.savefig(f'{output_folder}/{METRIC_TO_PLOT}_kl_div.png'.format(), bbox_inches='tight', pad_inches=0)

    # Ratio dist. plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    Y = [y for _, y in sorted(zip(ratio_dists, data_to_plot))]
    plt.plot(sorted(ratio_dists), Y)
    plt.scatter(sorted(ratio_dists), Y)
    plt.yscale('log')
    plt.xlabel('Ratio dist.')
    plt.ylabel(METRIC_TO_PLOT)
    plt.savefig(f'{output_folder}/{METRIC_TO_PLOT}_ratio_dist.png'.format(), bbox_inches='tight', pad_inches=0)

    # Wass. dist. plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    Y = [y for _, y in sorted(zip(wass_dists, data_to_plot))]
    plt.plot(sorted(wass_dists), Y)
    plt.scatter(sorted(wass_dists), Y)
    plt.yscale('log')
    plt.xlabel('Wass. dist.')
    plt.ylabel(METRIC_TO_PLOT)
    plt.savefig(f'{output_folder}/{METRIC_TO_PLOT}_wass_dist.png'.format(), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
