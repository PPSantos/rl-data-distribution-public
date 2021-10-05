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

#################################################################
OPTIMAL_SAMPLING_DIST_IDS = ['gridEnv4_sampling_dist_2021-10-05-10-52-48', 'gridEnv4_sampling_dist_2021-10-05-10-52-50', 'gridEnv4_sampling_dist_2021-10-05-10-52-53', 'gridEnv4_sampling_dist_2021-10-05-10-52-56', 'gridEnv4_sampling_dist_2021-10-05-10-52-59', 'gridEnv4_sampling_dist_2021-10-05-10-53-01', 'gridEnv4_sampling_dist_2021-10-05-10-53-04', 'gridEnv4_sampling_dist_2021-10-05-10-53-07', 'gridEnv4_sampling_dist_2021-10-05-10-53-09', 'gridEnv4_sampling_dist_2021-10-05-10-53-12', 'gridEnv4_sampling_dist_2021-10-05-10-53-15', 'gridEnv4_sampling_dist_2021-10-05-10-53-18', 'gridEnv4_sampling_dist_2021-10-05-10-53-20', 'gridEnv4_sampling_dist_2021-10-05-10-53-23', 'gridEnv4_sampling_dist_2021-10-05-10-53-26', 'gridEnv4_sampling_dist_2021-10-05-10-53-28', 'gridEnv4_sampling_dist_2021-10-05-10-53-31', 'gridEnv4_sampling_dist_2021-10-05-10-53-34', 'gridEnv4_sampling_dist_2021-10-05-10-53-37', 'gridEnv4_sampling_dist_2021-10-05-10-53-39']

SAMPLING_DISTS_IDS = ['gridEnv4_sampling_dist_2021-10-05-00-21-18', 'gridEnv4_sampling_dist_2021-10-05-00-32-27', 'gridEnv4_sampling_dist_2021-10-05-00-43-46', 'gridEnv4_sampling_dist_2021-10-05-00-55-01', 'gridEnv4_sampling_dist_2021-10-05-01-06-15', 'gridEnv4_sampling_dist_2021-10-05-01-17-29', 'gridEnv4_sampling_dist_2021-10-05-01-28-42', 'gridEnv4_sampling_dist_2021-10-05-01-40-01', 'gridEnv4_sampling_dist_2021-10-05-01-51-14', 'gridEnv4_sampling_dist_2021-10-05-02-02-29', 'gridEnv4_sampling_dist_2021-10-05-02-13-40', 'gridEnv4_sampling_dist_2021-10-05-02-24-54', 'gridEnv4_sampling_dist_2021-10-05-02-36-06', 'gridEnv4_sampling_dist_2021-10-05-02-47-14', 'gridEnv4_sampling_dist_2021-10-05-02-58-27', 'gridEnv4_sampling_dist_2021-10-05-03-09-38', 'gridEnv4_sampling_dist_2021-10-05-03-20-55', 'gridEnv4_sampling_dist_2021-10-05-03-32-11', 'gridEnv4_sampling_dist_2021-10-05-03-43-24', 'gridEnv4_sampling_dist_2021-10-05-03-54-36', 'gridEnv4_sampling_dist_2021-10-05-04-05-51', 'gridEnv4_sampling_dist_2021-10-05-04-17-02', 'gridEnv4_sampling_dist_2021-10-05-04-28-16', 'gridEnv4_sampling_dist_2021-10-05-04-39-24', 'gridEnv4_sampling_dist_2021-10-05-04-50-33', 'gridEnv4_sampling_dist_2021-10-05-05-01-47', 'gridEnv4_sampling_dist_2021-10-05-05-12-57', 'gridEnv4_sampling_dist_2021-10-05-05-24-08', 'gridEnv4_sampling_dist_2021-10-05-05-35-25', 'gridEnv4_sampling_dist_2021-10-05-05-46-40', 'gridEnv4_sampling_dist_2021-10-05-05-57-56', 'gridEnv4_sampling_dist_2021-10-05-06-09-11', 'gridEnv4_sampling_dist_2021-10-05-06-20-22', 'gridEnv4_sampling_dist_2021-10-05-06-31-32', 'gridEnv4_sampling_dist_2021-10-05-06-42-49', 'gridEnv4_sampling_dist_2021-10-05-06-54-02', 'gridEnv4_sampling_dist_2021-10-05-07-05-12', 'gridEnv4_sampling_dist_2021-10-05-07-16-20', 'gridEnv4_sampling_dist_2021-10-05-07-27-37', 'gridEnv4_sampling_dist_2021-10-05-07-38-52', 'gridEnv4_sampling_dist_2021-10-05-07-50-04', 'gridEnv4_sampling_dist_2021-10-05-08-01-16', 'gridEnv4_sampling_dist_2021-10-05-08-12-28', 'gridEnv4_sampling_dist_2021-10-05-08-23-46', 'gridEnv4_sampling_dist_2021-10-05-08-35-02', 'gridEnv4_sampling_dist_2021-10-05-08-46-15', 'gridEnv4_sampling_dist_2021-10-05-08-57-26', 'gridEnv4_sampling_dist_2021-10-05-09-08-42', 'gridEnv4_sampling_dist_2021-10-05-09-19-52', 'gridEnv4_sampling_dist_2021-10-05-09-32-07']

OFFLINE_DQN_EXP_IDS = ['gridEnv4_offline_dqn_2021-10-05-00-21-27', 'gridEnv4_offline_dqn_2021-10-05-00-32-36', 'gridEnv4_offline_dqn_2021-10-05-00-43-54', 'gridEnv4_offline_dqn_2021-10-05-00-55-09', 'gridEnv4_offline_dqn_2021-10-05-01-06-24', 'gridEnv4_offline_dqn_2021-10-05-01-17-38', 'gridEnv4_offline_dqn_2021-10-05-01-28-51', 'gridEnv4_offline_dqn_2021-10-05-01-40-10', 'gridEnv4_offline_dqn_2021-10-05-01-51-22', 'gridEnv4_offline_dqn_2021-10-05-02-02-37', 'gridEnv4_offline_dqn_2021-10-05-02-13-49', 'gridEnv4_offline_dqn_2021-10-05-02-25-03', 'gridEnv4_offline_dqn_2021-10-05-02-36-15', 'gridEnv4_offline_dqn_2021-10-05-02-47-23', 'gridEnv4_offline_dqn_2021-10-05-02-58-36', 'gridEnv4_offline_dqn_2021-10-05-03-09-47', 'gridEnv4_offline_dqn_2021-10-05-03-21-04', 'gridEnv4_offline_dqn_2021-10-05-03-32-20', 'gridEnv4_offline_dqn_2021-10-05-03-43-33', 'gridEnv4_offline_dqn_2021-10-05-03-54-45', 'gridEnv4_offline_dqn_2021-10-05-04-05-59', 'gridEnv4_offline_dqn_2021-10-05-04-17-11', 'gridEnv4_offline_dqn_2021-10-05-04-28-24', 'gridEnv4_offline_dqn_2021-10-05-04-39-32', 'gridEnv4_offline_dqn_2021-10-05-04-50-41', 'gridEnv4_offline_dqn_2021-10-05-05-01-55', 'gridEnv4_offline_dqn_2021-10-05-05-13-06', 'gridEnv4_offline_dqn_2021-10-05-05-24-17', 'gridEnv4_offline_dqn_2021-10-05-05-35-34', 'gridEnv4_offline_dqn_2021-10-05-05-46-49', 'gridEnv4_offline_dqn_2021-10-05-05-58-04', 'gridEnv4_offline_dqn_2021-10-05-06-09-19', 'gridEnv4_offline_dqn_2021-10-05-06-20-31', 'gridEnv4_offline_dqn_2021-10-05-06-31-41', 'gridEnv4_offline_dqn_2021-10-05-06-42-57', 'gridEnv4_offline_dqn_2021-10-05-06-54-11', 'gridEnv4_offline_dqn_2021-10-05-07-05-21', 'gridEnv4_offline_dqn_2021-10-05-07-16-28', 'gridEnv4_offline_dqn_2021-10-05-07-27-46', 'gridEnv4_offline_dqn_2021-10-05-07-39-01', 'gridEnv4_offline_dqn_2021-10-05-07-50-12', 'gridEnv4_offline_dqn_2021-10-05-08-01-24', 'gridEnv4_offline_dqn_2021-10-05-08-12-36', 'gridEnv4_offline_dqn_2021-10-05-08-23-54', 'gridEnv4_offline_dqn_2021-10-05-08-35-10', 'gridEnv4_offline_dqn_2021-10-05-08-46-24', 'gridEnv4_offline_dqn_2021-10-05-08-57-34', 'gridEnv4_offline_dqn_2021-10-05-09-08-51', 'gridEnv4_offline_dqn_2021-10-05-09-20-01', 'gridEnv4_offline_dqn_2021-10-05-09-32-15']
#################################################################

FIGURE_X = 8.0
FIGURE_Y = 6.0
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'

def main():
    # Prepare output folder.
    output_folder = PLOTS_FOLDER_PATH + '/distances_plots'
    os.makedirs(output_folder, exist_ok=True)

    optimal_dists = []
    for optimal_dist_id in OPTIMAL_SAMPLING_DIST_IDS:
        optimal_dist_path = DATA_FOLDER_PATH + optimal_dist_id + '/data.json'
        print(optimal_dist_path)
        with open(optimal_dist_path, 'r') as f:
            data = json.load(f)
            d = np.array(json.loads(data)['sampling_dist'])
        f.close()
        optimal_dists.append(d)
    print('len(optimal_dists)', len(optimal_dists))

    sampling_dists = []
    for sampling_dist_id in SAMPLING_DISTS_IDS:
        sampling_dist_path = DATA_FOLDER_PATH + sampling_dist_id + '/data.json'
        with open(sampling_dist_path, 'r') as f:
            data = json.load(f)
            d = np.array(json.loads(data)['sampling_dist'])
        f.close()
        sampling_dists.append(d)
    print('len(sampling_dists)', len(sampling_dists))

    offline_dqn_metrics = []
    for offline_dqn_exp_id in OFFLINE_DQN_EXP_IDS:
        exp_metrics_path = PLOTS_FOLDER_PATH + offline_dqn_exp_id + '/scalar_metrics.json'
        with open(exp_metrics_path, 'r') as f:
            d = json.load(f)
        f.close()
        offline_dqn_metrics.append(d)
    print('len(offline_dqn_metrics)', len(offline_dqn_metrics))

    wass_dists = []
    ratio_dists = []
    kl_dists = []
    for sampling_dist in sampling_dists:

        wass_dist = np.min([scipy.stats.wasserstein_distance(optimal_dist, sampling_dist)
                        for optimal_dist in optimal_dists])
        ratio_dist = np.min([np.max(optimal_dist/(sampling_dist+1e-06))
                        for optimal_dist in optimal_dists])
        kl_dist = np.min([scipy.stats.entropy(optimal_dist,sampling_dist+1e-06)
                        for optimal_dist in optimal_dists])

        wass_dists.append(wass_dist)
        ratio_dists.append(ratio_dist)
        kl_dists.append(kl_dist)

    print('wass_dists', wass_dists)
    print('ratio_dists', ratio_dists)
    print('kl_dists', kl_dists)

    for metric in ['qvals_avg_error', 'rewards_default']:

        data_to_plot = [x[metric] for x in offline_dqn_metrics]

        # KL-div. plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        Y = [y for _, y in sorted(zip(kl_dists, data_to_plot))]
        #plt.plot(sorted(kl_dists), Y)
        plt.scatter(sorted(kl_dists), Y)
        plt.xlabel('KL div.')
        plt.yscale('log')
        plt.ylabel(metric)
        if metric == 'qvals_avg_error':
            plt.yscale('log')
        plt.savefig(f'{output_folder}/{metric}_kl_div.png'.format(), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/{metric}_kl_div.pdf'.format(), bbox_inches='tight', pad_inches=0)

        # Ratio dist. plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        Y = [y for _, y in sorted(zip(ratio_dists, data_to_plot))]
        #plt.plot(sorted(ratio_dists), Y)
        plt.scatter(sorted(ratio_dists), Y)
        plt.yscale('log')
        plt.xlabel('Ratio dist.')
        plt.ylabel(metric)
        if metric == 'qvals_avg_error':
            plt.yscale('log')
        plt.savefig(f'{output_folder}/{metric}_ratio_dist.png'.format(), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/{metric}_ratio_dist.pdf'.format(), bbox_inches='tight', pad_inches=0)

        # Wasserstein dist. plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        Y = [y for _, y in sorted(zip(wass_dists, data_to_plot))]
        #plt.plot(sorted(wass_dists), Y)
        plt.scatter(sorted(wass_dists), Y)
        plt.yscale('log')
        plt.xlabel('Wass. dist.')
        plt.ylabel(metric)
        if metric == 'qvals_avg_error':
            plt.yscale('log')
        plt.savefig(f'{output_folder}/{metric}_wass_dist.png'.format(), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/{metric}_wass_dist.pdf'.format(), bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
