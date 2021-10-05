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
OPTIMAL_SAMPLING_DIST_IDS = [

]

SAMPLING_DISTS_IDS = [

]

OFFLINE_DQN_EXP_IDS = [

]
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
    for optimal_dist_id in OPTIMAL_DISTS_IDS:
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
        kl_dist = np.min([scipy.stats.entropy(optimal_dist,sampling_dist)
                        for optimal_dist in optimal_dists])

        wass_dists.append(wass_dist)
        ratio_dists.append(ratio_dist)
        kl_dists.append(kl_dist)

    print('wass_dists', wass_dists)
    print('ratio_dists', ratio_dists)
    print('kl_dists', kl_dists)
    
    for metric in ['qvals_avg_error', 'default_rewards']:

        data_to_plot = [x[metric] for x in offline_dqn_metrics]

        # KL-div. plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        Y = [y for _, y in sorted(zip(kl_dists, data_to_plot))]
        plt.plot(sorted(kl_dists), Y)
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
        plt.plot(sorted(ratio_dists), Y)
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
        plt.plot(sorted(wass_dists), Y)
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
