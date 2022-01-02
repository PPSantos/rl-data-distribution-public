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
'multiPathsEnv_sampling_dist_2021-10-07-00-49-56', 'multiPathsEnv_sampling_dist_2021-10-07-00-49-58', 'multiPathsEnv_sampling_dist_2021-10-07-00-49-59', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-01', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-03', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-04', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-06', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-07', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-09', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-11', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-12', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-14', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-15', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-17', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-19', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-20', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-22', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-24', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-25', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-27',
]
OFFLINE_DQN_EXP_IDS = [
'multiPathsEnv_offline_dqn_2021-10-06-19-56-37', 'multiPathsEnv_offline_dqn_2021-10-06-20-03-46', 'multiPathsEnv_offline_dqn_2021-10-06-20-10-53', 'multiPathsEnv_offline_dqn_2021-10-06-20-18-08', 'multiPathsEnv_offline_dqn_2021-10-06-20-25-24', 'multiPathsEnv_offline_dqn_2021-10-06-20-33-11', 'multiPathsEnv_offline_dqn_2021-10-06-20-41-57', 'multiPathsEnv_offline_dqn_2021-10-06-20-50-34', 'multiPathsEnv_offline_dqn_2021-10-06-20-59-13', 'multiPathsEnv_offline_dqn_2021-10-06-21-08-00', 'multiPathsEnv_offline_dqn_2021-10-06-21-16-54', 'multiPathsEnv_offline_dqn_2021-10-06-21-25-29', 'multiPathsEnv_offline_dqn_2021-10-06-21-34-07', 'multiPathsEnv_offline_dqn_2021-10-06-21-42-46', 'multiPathsEnv_offline_dqn_2021-10-06-21-51-44', 'multiPathsEnv_offline_dqn_2021-10-06-22-00-21', 'multiPathsEnv_offline_dqn_2021-10-06-22-08-58', 'multiPathsEnv_offline_dqn_2021-10-06-22-17-33', 'multiPathsEnv_offline_dqn_2021-10-06-22-26-37', 'multiPathsEnv_offline_dqn_2021-10-06-22-35-19', 'multiPathsEnv_offline_dqn_2021-10-06-22-43-52', 'multiPathsEnv_offline_dqn_2021-10-06-22-52-31', 'multiPathsEnv_offline_dqn_2021-10-06-23-01-23', 'multiPathsEnv_offline_dqn_2021-10-06-23-10-10', 'multiPathsEnv_offline_dqn_2021-10-06-23-18-44',
'multiPathsEnv_offline_dqn_2021-10-10-14-31-58', 'multiPathsEnv_offline_dqn_2021-10-10-14-43-35', 'multiPathsEnv_offline_dqn_2021-10-10-14-55-59', 'multiPathsEnv_offline_dqn_2021-10-10-15-07-12', 'multiPathsEnv_offline_dqn_2021-10-10-15-21-24', 'multiPathsEnv_offline_dqn_2021-10-10-15-33-08', 'multiPathsEnv_offline_dqn_2021-10-10-15-47-56', 'multiPathsEnv_offline_dqn_2021-10-10-16-00-01', 'multiPathsEnv_offline_dqn_2021-10-10-16-12-46', 'multiPathsEnv_offline_dqn_2021-10-10-16-26-40', 'multiPathsEnv_offline_dqn_2021-10-10-16-39-24', 'multiPathsEnv_offline_dqn_2021-10-10-16-53-06', 'multiPathsEnv_offline_dqn_2021-10-10-17-11-00', 'multiPathsEnv_offline_dqn_2021-10-10-17-22-59', 'multiPathsEnv_offline_dqn_2021-10-10-17-37-00', 'multiPathsEnv_offline_dqn_2021-10-10-17-48-32', 'multiPathsEnv_offline_dqn_2021-10-10-18-00-54', 'multiPathsEnv_offline_dqn_2021-10-10-18-15-23', 'multiPathsEnv_offline_dqn_2021-10-10-18-27-55', 'multiPathsEnv_offline_dqn_2021-10-10-18-42-42', 'multiPathsEnv_offline_dqn_2021-10-10-18-58-31', 'multiPathsEnv_offline_dqn_2021-10-10-19-10-24', 'multiPathsEnv_offline_dqn_2021-10-10-19-22-33', 'multiPathsEnv_offline_dqn_2021-10-10-19-36-48', 'multiPathsEnv_offline_dqn_2021-10-10-19-48-34',
'multiPathsEnv_offline_dqn_2021-10-11-18-06-15', 'multiPathsEnv_offline_dqn_2021-10-11-18-59-51', 'multiPathsEnv_offline_dqn_2021-10-11-19-45-27', 'multiPathsEnv_offline_dqn_2021-10-11-20-27-45', 'multiPathsEnv_offline_dqn_2021-10-11-21-10-34', 'multiPathsEnv_offline_dqn_2021-10-11-21-53-43', 'multiPathsEnv_offline_dqn_2021-10-11-22-38-45', 'multiPathsEnv_offline_dqn_2021-10-11-23-23-11', 'multiPathsEnv_offline_dqn_2021-10-12-00-16-06', 'multiPathsEnv_offline_dqn_2021-10-12-01-14-30', 'multiPathsEnv_offline_dqn_2021-10-12-01-57-54', 'multiPathsEnv_offline_dqn_2021-10-12-02-49-02', 'multiPathsEnv_offline_dqn_2021-10-12-03-31-53', 'multiPathsEnv_offline_dqn_2021-10-12-04-17-08', 'multiPathsEnv_offline_dqn_2021-10-12-04-59-58', 'multiPathsEnv_offline_dqn_2021-10-12-05-43-03', 'multiPathsEnv_offline_dqn_2021-10-12-06-34-28', 'multiPathsEnv_offline_dqn_2021-10-12-07-29-35', 'multiPathsEnv_offline_dqn_2021-10-12-08-12-13', 'multiPathsEnv_offline_dqn_2021-10-12-09-03-15', 'multiPathsEnv_offline_dqn_2021-10-12-09-46-17', 'multiPathsEnv_offline_dqn_2021-10-12-10-37-22',
]
SAMPLING_DISTS_IDS = [
'multiPathsEnv_sampling_dist_2021-10-06-19-56-33', 'multiPathsEnv_sampling_dist_2021-10-06-20-03-42', 'multiPathsEnv_sampling_dist_2021-10-06-20-10-48', 'multiPathsEnv_sampling_dist_2021-10-06-20-18-04', 'multiPathsEnv_sampling_dist_2021-10-06-20-25-20', 'multiPathsEnv_sampling_dist_2021-10-06-20-33-06', 'multiPathsEnv_sampling_dist_2021-10-06-20-41-52', 'multiPathsEnv_sampling_dist_2021-10-06-20-50-30', 'multiPathsEnv_sampling_dist_2021-10-06-20-59-08', 'multiPathsEnv_sampling_dist_2021-10-06-21-07-54', 'multiPathsEnv_sampling_dist_2021-10-06-21-16-49', 'multiPathsEnv_sampling_dist_2021-10-06-21-25-25', 'multiPathsEnv_sampling_dist_2021-10-06-21-34-03', 'multiPathsEnv_sampling_dist_2021-10-06-21-42-41', 'multiPathsEnv_sampling_dist_2021-10-06-21-51-39', 'multiPathsEnv_sampling_dist_2021-10-06-22-00-16', 'multiPathsEnv_sampling_dist_2021-10-06-22-08-53', 'multiPathsEnv_sampling_dist_2021-10-06-22-17-28', 'multiPathsEnv_sampling_dist_2021-10-06-22-26-31', 'multiPathsEnv_sampling_dist_2021-10-06-22-35-14', 'multiPathsEnv_sampling_dist_2021-10-06-22-43-48', 'multiPathsEnv_sampling_dist_2021-10-06-22-52-26', 'multiPathsEnv_sampling_dist_2021-10-06-23-01-18', 'multiPathsEnv_sampling_dist_2021-10-06-23-10-05', 'multiPathsEnv_sampling_dist_2021-10-06-23-18-39',
'multiPathsEnv_sampling_dist_2021-10-10-14-31-52', 'multiPathsEnv_sampling_dist_2021-10-10-14-43-30', 'multiPathsEnv_sampling_dist_2021-10-10-14-55-54', 'multiPathsEnv_sampling_dist_2021-10-10-15-07-06', 'multiPathsEnv_sampling_dist_2021-10-10-15-21-19', 'multiPathsEnv_sampling_dist_2021-10-10-15-33-02', 'multiPathsEnv_sampling_dist_2021-10-10-15-47-50', 'multiPathsEnv_sampling_dist_2021-10-10-15-59-56', 'multiPathsEnv_sampling_dist_2021-10-10-16-12-41', 'multiPathsEnv_sampling_dist_2021-10-10-16-26-34', 'multiPathsEnv_sampling_dist_2021-10-10-16-39-18', 'multiPathsEnv_sampling_dist_2021-10-10-16-53-01', 'multiPathsEnv_sampling_dist_2021-10-10-17-10-55', 'multiPathsEnv_sampling_dist_2021-10-10-17-22-54', 'multiPathsEnv_sampling_dist_2021-10-10-17-36-55', 'multiPathsEnv_sampling_dist_2021-10-10-17-48-27', 'multiPathsEnv_sampling_dist_2021-10-10-18-00-48', 'multiPathsEnv_sampling_dist_2021-10-10-18-15-17', 'multiPathsEnv_sampling_dist_2021-10-10-18-27-50', 'multiPathsEnv_sampling_dist_2021-10-10-18-42-37', 'multiPathsEnv_sampling_dist_2021-10-10-18-58-26', 'multiPathsEnv_sampling_dist_2021-10-10-19-10-19', 'multiPathsEnv_sampling_dist_2021-10-10-19-22-27', 'multiPathsEnv_sampling_dist_2021-10-10-19-36-43', 'multiPathsEnv_sampling_dist_2021-10-10-19-48-29',
'multiPathsEnv_sampling_dist_2021-10-11-18-06-10', 'multiPathsEnv_sampling_dist_2021-10-11-18-59-46', 'multiPathsEnv_sampling_dist_2021-10-11-19-45-22', 'multiPathsEnv_sampling_dist_2021-10-11-20-27-40', 'multiPathsEnv_sampling_dist_2021-10-11-21-10-29', 'multiPathsEnv_sampling_dist_2021-10-11-21-53-38', 'multiPathsEnv_sampling_dist_2021-10-11-22-38-39', 'multiPathsEnv_sampling_dist_2021-10-11-23-23-06', 'multiPathsEnv_sampling_dist_2021-10-12-00-16-00', 'multiPathsEnv_sampling_dist_2021-10-12-01-14-24', 'multiPathsEnv_sampling_dist_2021-10-12-01-57-48', 'multiPathsEnv_sampling_dist_2021-10-12-02-48-57', 'multiPathsEnv_sampling_dist_2021-10-12-03-31-47', 'multiPathsEnv_sampling_dist_2021-10-12-04-17-03', 'multiPathsEnv_sampling_dist_2021-10-12-04-59-52', 'multiPathsEnv_sampling_dist_2021-10-12-05-42-57', 'multiPathsEnv_sampling_dist_2021-10-12-06-34-23', 'multiPathsEnv_sampling_dist_2021-10-12-07-29-30', 'multiPathsEnv_sampling_dist_2021-10-12-08-12-08', 'multiPathsEnv_sampling_dist_2021-10-12-09-03-09', 'multiPathsEnv_sampling_dist_2021-10-12-09-46-11', 'multiPathsEnv_sampling_dist_2021-10-12-10-37-17',
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
        sm_x, sm_y = lowess(Y, X,  frac=0.6, 
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
