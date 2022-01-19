import os
import json
import numpy as np
import pathlib
import tarfile
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
OPTIMAL_SAMPLING_DIST_IDS = None

EXP_IDS = [
#'gridEnv1_offline_dqn_2022-01-18-22-07-14', 'gridEnv1_offline_dqn_2022-01-18-22-15-26', 'gridEnv1_offline_dqn_2022-01-18-22-23-08', 'gridEnv1_offline_dqn_2022-01-18-22-31-01', 'gridEnv1_offline_dqn_2022-01-18-22-38-56', 'gridEnv1_offline_dqn_2022-01-18-22-46-39', 'gridEnv1_offline_dqn_2022-01-18-22-54-30', 'gridEnv1_offline_dqn_2022-01-18-23-02-12', 'gridEnv1_offline_dqn_2022-01-18-23-09-59', 'gridEnv1_offline_dqn_2022-01-18-23-17-44', 'gridEnv1_offline_dqn_2022-01-18-23-25-34', 'gridEnv1_offline_dqn_2022-01-18-23-33-21', 'gridEnv1_offline_dqn_2022-01-18-23-41-07', 'gridEnv1_offline_dqn_2022-01-18-23-49-02',
'gridEnv1_offline_dqn_2022-01-19-16-05-47', 'gridEnv1_offline_dqn_2022-01-19-16-14-01', 'gridEnv1_offline_dqn_2022-01-19-16-21-57', 'gridEnv1_offline_dqn_2022-01-19-16-29-46', 'gridEnv1_offline_dqn_2022-01-19-16-37-38', 'gridEnv1_offline_dqn_2022-01-19-16-45-32', 'gridEnv1_offline_dqn_2022-01-19-16-53-46', 'gridEnv1_offline_dqn_2022-01-19-17-01-37', 'gridEnv1_offline_dqn_2022-01-19-17-09-23', 'gridEnv1_offline_dqn_2022-01-19-17-17-07', 'gridEnv1_offline_dqn_2022-01-19-17-25-02', 'gridEnv1_offline_dqn_2022-01-19-17-32-51', 'gridEnv1_offline_dqn_2022-01-19-17-40-41', 'gridEnv1_offline_dqn_2022-01-19-17-48-27', 'gridEnv1_offline_dqn_2022-01-19-17-56-19', 'gridEnv1_offline_dqn_2022-01-19-18-04-20', 'gridEnv1_offline_dqn_2022-01-19-18-12-08', 'gridEnv1_offline_dqn_2022-01-19-18-19-59', 'gridEnv1_offline_dqn_2022-01-19-18-27-43', 'gridEnv1_offline_dqn_2022-01-19-18-35-27', 'gridEnv1_offline_dqn_2022-01-19-18-43-09', 'gridEnv1_offline_dqn_2022-01-19-18-50-57',
]

#################################################################

FIGURE_X = 6.0
FIGURE_Y = 4.0
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'

def main():

    # Prepare output folder.
    output_folder = PLOTS_FOLDER_PATH + '/plots_env'
    os.makedirs(output_folder, exist_ok=True)

    # optimal_dists = []
    # for optimal_dist_id in OPTIMAL_SAMPLING_DIST_IDS:
    #     optimal_dist_path = DATA_FOLDER_PATH + optimal_dist_id + '/data.json'
    #     print(optimal_dist_path)
    #     with open(optimal_dist_path, 'r') as f:
    #         data = json.load(f)
    #         d = np.array(json.loads(data)['sampling_dist'])
    #     f.close()
    #     optimal_dists.append(d)
    # print('len(optimal_dists)', len(optimal_dists))

    # 
    metrics = []
    for offline_exp_id in EXP_IDS:
        exp_metrics_path = PLOTS_FOLDER_PATH + offline_exp_id + '/scalar_metrics.json'
        with open(exp_metrics_path, 'r') as f:
            d = json.load(f)
        f.close()
        metrics.append(d)
    print('len(metrics)', len(metrics))
    print(metrics[0].keys())

    dataset_metrics = []
    for offline_exp_id in EXP_IDS:
        exp_folder_path = DATA_FOLDER_PATH + offline_exp_id + '.tar.gz'

        tar = tarfile.open(exp_folder_path)
        data_file = tar.extractfile("{0}/dataset_info.json".format(offline_exp_id))

        dataset_info = json.load(data_file)
        dataset_info = json.loads(dataset_info)

        dataset_metrics.append(dataset_info)

    print('len(dataset_metrics)', len(dataset_metrics))
    print(dataset_metrics[0].keys())

    # Entropy plots.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = [x['dataset_entropy'] for x in dataset_metrics]
    Y = [y['qvals_avg_error'] for y in metrics]

    plt.scatter(X, Y)

    plt.yscale('log')

    plt.xlabel('Entropy')
    plt.ylabel('Q-value error')

    plt.savefig(f'{output_folder}/entropy_error.png'.format(), bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'{output_folder}/entropy_error.pdf'.format(), bbox_inches='tight', pad_inches=0) 

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = [x['dataset_entropy'] for x in dataset_metrics]
    Y = [y['rollouts_rewards_final'] for y in metrics]

    plt.scatter(X, Y)

    plt.xlabel('Entropy')
    plt.ylabel('Average reward')

    plt.savefig(f'{output_folder}/entropy_reward.png'.format(), bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'{output_folder}/entropy_reward.pdf'.format(), bbox_inches='tight', pad_inches=0) 

    exit()


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
