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
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

#################################################################
OPTIMAL_SAMPLING_DIST_IDS = [
'pendulum_sampling_dist_2021-10-07-18-40-46', 'pendulum_sampling_dist_2021-10-07-18-40-51', 'pendulum_sampling_dist_2021-10-07-18-40-57', 'pendulum_sampling_dist_2021-10-07-18-41-03', 'pendulum_sampling_dist_2021-10-07-18-41-09', 'pendulum_sampling_dist_2021-10-07-18-41-14', 'pendulum_sampling_dist_2021-10-07-18-41-20', 'pendulum_sampling_dist_2021-10-07-18-41-26', 'pendulum_sampling_dist_2021-10-07-18-41-32', 'pendulum_sampling_dist_2021-10-07-18-41-37', 'pendulum_sampling_dist_2021-10-07-18-41-43', 'pendulum_sampling_dist_2021-10-07-18-41-49', 'pendulum_sampling_dist_2021-10-07-18-41-55', 'pendulum_sampling_dist_2021-10-07-18-42-00', 'pendulum_sampling_dist_2021-10-07-18-42-06', 'pendulum_sampling_dist_2021-10-07-18-42-12', 'pendulum_sampling_dist_2021-10-07-18-42-18', 'pendulum_sampling_dist_2021-10-07-18-42-23', 'pendulum_sampling_dist_2021-10-07-18-42-29', 'pendulum_sampling_dist_2021-10-07-18-42-35'
]
OFFLINE_DQN_EXP_IDS = [
'pendulum_offline_dqn_2021-10-07-09-55-12', 'pendulum_offline_dqn_2021-10-07-10-09-51', 'pendulum_offline_dqn_2021-10-07-10-24-12', 'pendulum_offline_dqn_2021-10-07-10-38-38', 'pendulum_offline_dqn_2021-10-07-10-53-21', 'pendulum_offline_dqn_2021-10-07-11-07-51', 'pendulum_offline_dqn_2021-10-07-11-22-26', 'pendulum_offline_dqn_2021-10-07-11-39-56', 'pendulum_offline_dqn_2021-10-07-11-56-36', 'pendulum_offline_dqn_2021-10-07-12-13-42', 'pendulum_offline_dqn_2021-10-07-12-28-57', 'pendulum_offline_dqn_2021-10-07-12-44-44', 'pendulum_offline_dqn_2021-10-07-13-01-25', 'pendulum_offline_dqn_2021-10-07-13-19-19', 'pendulum_offline_dqn_2021-10-07-13-35-27', 'pendulum_offline_dqn_2021-10-07-13-50-17', 'pendulum_offline_dqn_2021-10-07-14-04-50', 'pendulum_offline_dqn_2021-10-07-14-20-02', 'pendulum_offline_dqn_2021-10-07-14-36-46', 'pendulum_offline_dqn_2021-10-07-14-52-42', 'pendulum_offline_dqn_2021-10-07-15-09-27', 'pendulum_offline_dqn_2021-10-07-15-26-29', 'pendulum_offline_dqn_2021-10-07-15-43-12', 'pendulum_offline_dqn_2021-10-07-16-00-20', 'pendulum_offline_dqn_2021-10-07-16-16-37', 'pendulum_offline_dqn_2021-10-07-16-31-11', 'pendulum_offline_dqn_2021-10-07-16-48-12', 'pendulum_offline_dqn_2021-10-07-17-05-35', 'pendulum_offline_dqn_2021-10-07-17-22-12', 'pendulum_offline_dqn_2021-10-07-17-38-49',
'pendulum_offline_dqn_2021-10-07-18-57-22', 'pendulum_offline_dqn_2021-10-07-19-15-22', 'pendulum_offline_dqn_2021-10-07-19-32-08', 'pendulum_offline_dqn_2021-10-07-19-48-33', 'pendulum_offline_dqn_2021-10-07-20-05-14', 'pendulum_offline_dqn_2021-10-07-20-21-47', 'pendulum_offline_dqn_2021-10-07-20-36-46', 'pendulum_offline_dqn_2021-10-07-20-51-14', 'pendulum_offline_dqn_2021-10-07-21-05-49', 'pendulum_offline_dqn_2021-10-07-21-21-35', 'pendulum_offline_dqn_2021-10-07-21-39-16', 'pendulum_offline_dqn_2021-10-07-21-56-18', 'pendulum_offline_dqn_2021-10-07-22-13-17', 'pendulum_offline_dqn_2021-10-07-22-31-00', 'pendulum_offline_dqn_2021-10-07-22-45-49', 'pendulum_offline_dqn_2021-10-07-23-00-26', 'pendulum_offline_dqn_2021-10-07-23-15-25', 'pendulum_offline_dqn_2021-10-07-23-31-36', 'pendulum_offline_dqn_2021-10-07-23-46-32', 'pendulum_offline_dqn_2021-10-08-00-01-08', 'pendulum_offline_dqn_2021-10-08-00-15-44', 'pendulum_offline_dqn_2021-10-08-00-30-20', 'pendulum_offline_dqn_2021-10-08-00-46-15', 'pendulum_offline_dqn_2021-10-08-01-01-06', 'pendulum_offline_dqn_2021-10-08-01-16-22', 'pendulum_offline_dqn_2021-10-08-01-33-44', 'pendulum_offline_dqn_2021-10-08-01-49-16', 'pendulum_offline_dqn_2021-10-08-02-03-59', 'pendulum_offline_dqn_2021-10-08-02-18-30', 'pendulum_offline_dqn_2021-10-08-02-32-59',
'pendulum_offline_dqn_2021-10-08-12-12-44', 'pendulum_offline_dqn_2021-10-08-12-40-29', 'pendulum_offline_dqn_2021-10-08-13-07-06', 'pendulum_offline_dqn_2021-10-08-13-41-19', 'pendulum_offline_dqn_2021-10-08-13-58-05', 'pendulum_offline_dqn_2021-10-08-14-33-25', 'pendulum_offline_dqn_2021-10-08-14-52-25', 'pendulum_offline_dqn_2021-10-08-15-16-57', 'pendulum_offline_dqn_2021-10-08-15-33-13', 'pendulum_offline_dqn_2021-10-08-15-50-59',
'pendulum_offline_dqn_2021-10-10-11-18-55', 'pendulum_offline_dqn_2021-10-10-11-35-46', 'pendulum_offline_dqn_2021-10-10-11-51-51', 'pendulum_offline_dqn_2021-10-10-12-08-12', 'pendulum_offline_dqn_2021-10-10-12-23-28', 'pendulum_offline_dqn_2021-10-10-12-38-33', 'pendulum_offline_dqn_2021-10-10-12-53-30', 'pendulum_offline_dqn_2021-10-10-13-08-36', 'pendulum_offline_dqn_2021-10-10-13-23-23', 'pendulum_offline_dqn_2021-10-10-13-38-36',
]
SAMPLING_DISTS_IDS = [
'pendulum_sampling_dist_2021-10-07-09-54-52', 'pendulum_sampling_dist_2021-10-07-10-09-30', 'pendulum_sampling_dist_2021-10-07-10-23-51', 'pendulum_sampling_dist_2021-10-07-10-38-17', 'pendulum_sampling_dist_2021-10-07-10-53-00', 'pendulum_sampling_dist_2021-10-07-11-07-30', 'pendulum_sampling_dist_2021-10-07-11-22-04', 'pendulum_sampling_dist_2021-10-07-11-39-34', 'pendulum_sampling_dist_2021-10-07-11-56-14', 'pendulum_sampling_dist_2021-10-07-12-13-21', 'pendulum_sampling_dist_2021-10-07-12-28-36', 'pendulum_sampling_dist_2021-10-07-12-44-22', 'pendulum_sampling_dist_2021-10-07-13-01-03', 'pendulum_sampling_dist_2021-10-07-13-18-57', 'pendulum_sampling_dist_2021-10-07-13-35-06', 'pendulum_sampling_dist_2021-10-07-13-49-56', 'pendulum_sampling_dist_2021-10-07-14-04-29', 'pendulum_sampling_dist_2021-10-07-14-19-40', 'pendulum_sampling_dist_2021-10-07-14-36-25', 'pendulum_sampling_dist_2021-10-07-14-52-20', 'pendulum_sampling_dist_2021-10-07-15-09-04', 'pendulum_sampling_dist_2021-10-07-15-26-07', 'pendulum_sampling_dist_2021-10-07-15-42-50', 'pendulum_sampling_dist_2021-10-07-15-59-58', 'pendulum_sampling_dist_2021-10-07-16-16-16', 'pendulum_sampling_dist_2021-10-07-16-30-50', 'pendulum_sampling_dist_2021-10-07-16-47-50', 'pendulum_sampling_dist_2021-10-07-17-05-12', 'pendulum_sampling_dist_2021-10-07-17-21-51', 'pendulum_sampling_dist_2021-10-07-17-38-27',
'pendulum_sampling_dist_2021-10-07-18-57-01', 'pendulum_sampling_dist_2021-10-07-19-15-01', 'pendulum_sampling_dist_2021-10-07-19-31-46', 'pendulum_sampling_dist_2021-10-07-19-48-12', 'pendulum_sampling_dist_2021-10-07-20-04-53', 'pendulum_sampling_dist_2021-10-07-20-21-26', 'pendulum_sampling_dist_2021-10-07-20-36-25', 'pendulum_sampling_dist_2021-10-07-20-50-53', 'pendulum_sampling_dist_2021-10-07-21-05-28', 'pendulum_sampling_dist_2021-10-07-21-21-13', 'pendulum_sampling_dist_2021-10-07-21-38-54', 'pendulum_sampling_dist_2021-10-07-21-55-57', 'pendulum_sampling_dist_2021-10-07-22-12-56', 'pendulum_sampling_dist_2021-10-07-22-30-39', 'pendulum_sampling_dist_2021-10-07-22-45-28', 'pendulum_sampling_dist_2021-10-07-23-00-05', 'pendulum_sampling_dist_2021-10-07-23-15-05', 'pendulum_sampling_dist_2021-10-07-23-31-14', 'pendulum_sampling_dist_2021-10-07-23-46-11', 'pendulum_sampling_dist_2021-10-08-00-00-47', 'pendulum_sampling_dist_2021-10-08-00-15-23', 'pendulum_sampling_dist_2021-10-08-00-29-59', 'pendulum_sampling_dist_2021-10-08-00-45-54', 'pendulum_sampling_dist_2021-10-08-01-00-46', 'pendulum_sampling_dist_2021-10-08-01-16-01', 'pendulum_sampling_dist_2021-10-08-01-33-23', 'pendulum_sampling_dist_2021-10-08-01-48-55', 'pendulum_sampling_dist_2021-10-08-02-03-38', 'pendulum_sampling_dist_2021-10-08-02-18-09', 'pendulum_sampling_dist_2021-10-08-02-32-38',
'pendulum_sampling_dist_2021-10-08-12-12-23', 'pendulum_sampling_dist_2021-10-08-12-40-08', 'pendulum_sampling_dist_2021-10-08-13-06-45', 'pendulum_sampling_dist_2021-10-08-13-40-57', 'pendulum_sampling_dist_2021-10-08-13-57-44', 'pendulum_sampling_dist_2021-10-08-14-33-03', 'pendulum_sampling_dist_2021-10-08-14-52-04', 'pendulum_sampling_dist_2021-10-08-15-16-36', 'pendulum_sampling_dist_2021-10-08-15-32-52', 'pendulum_sampling_dist_2021-10-08-15-50-38',
'pendulum_sampling_dist_2021-10-10-11-18-34', 'pendulum_sampling_dist_2021-10-10-11-35-25', 'pendulum_sampling_dist_2021-10-10-11-51-30', 'pendulum_sampling_dist_2021-10-10-12-07-51', 'pendulum_sampling_dist_2021-10-10-12-23-07', 'pendulum_sampling_dist_2021-10-10-12-38-12', 'pendulum_sampling_dist_2021-10-10-12-53-09', 'pendulum_sampling_dist_2021-10-10-13-08-15', 'pendulum_sampling_dist_2021-10-10-13-23-01', 'pendulum_sampling_dist_2021-10-10-13-38-15',
]
#################################################################

FIGURE_X = 6.0
FIGURE_Y = 4.0
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
        X = sorted(kl_dists)

        #plt.plot(X, Y)
        plt.scatter(X, Y)

        # LOWESS smoothing.
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm_x, sm_y = lowess(Y, X,  frac=0.4, 
                                it=5, return_sorted = True).T
        plt.plot(sm_x, sm_y, label='LOWESS')

        plt.xlabel('KL div.')
        plt.yscale('log')
        if metric == 'qvals_avg_error':
            plt.ylabel(r'$Q$-values error')
        else:
            plt.ylabel('Reward')
        if metric == 'qvals_avg_error':
            plt.yscale('log')
        else:
            plt.yscale('linear')
        plt.savefig(f'{output_folder}/{metric}_kl_div.png'.format(), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/{metric}_kl_div.pdf'.format(), bbox_inches='tight', pad_inches=0)

        # Ratio dist. plot.
        """ fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        Y = [y for _, y in sorted(zip(ratio_dists, data_to_plot))]
        #plt.plot(sorted(ratio_dists), Y)
        plt.scatter(sorted(ratio_dists), Y)
        plt.yscale('log')
        #plt.xlabel('Ratio dist.')
        plt.ylabel(metric)
        if metric == 'qvals_avg_error':
            plt.yscale('log')
        else:
            plt.yscale('linear')
        plt.savefig(f'{output_folder}/{metric}_ratio_dist.png'.format(), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/{metric}_ratio_dist.pdf'.format(), bbox_inches='tight', pad_inches=0)

        # Wasserstein dist. plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)
        Y = [y for _, y in sorted(zip(wass_dists, data_to_plot))]
        #plt.plot(sorted(wass_dists), Y)
        plt.scatter(sorted(wass_dists), Y)
        plt.yscale('log')
        #plt.xlabel('Wass. dist.')
        plt.ylabel(metric)
        if metric == 'qvals_avg_error':
            plt.yscale('log')
        else:
            plt.yscale('linear')
        plt.savefig(f'{output_folder}/{metric}_wass_dist.png'.format(), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/{metric}_wass_dist.pdf'.format(), bbox_inches='tight', pad_inches=0) """

if __name__ == "__main__":
    main()
