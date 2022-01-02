import os
import json
import tarfile
import numpy as np
from numpy.core.defchararray import lower, upper 
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

from envs import env_suite

#################################################################
ENV_ID_TO_PLOT = 'pendulum'
ENVS_LABELS = {
    'gridEnv1': 'Grid 1',
    'gridEnv4': 'Grid 2',
    'multiPathsEnv': 'Multi-path',
    'pendulum': 'Pendulum',
    'mountaincar': 'Mountain car',
}

UNIFORM_SAMPLING_DIST_ENTROPY = {
    'gridEnv1': 5.77,
    'gridEnv4': 5.63,
    'multiPathsEnv': 4.91,
    'pendulum': 8.54,
    'mountaincar': 8.59,
}

MAXIMUM_REWARD = {
    'gridEnv1': 36.0,
    'gridEnv4': 41.0,
    'multiPathsEnv': 5.0,
    'pendulum': 42.8, # TODO: check this.
    'mountaincar': 50.0, # TODO: check this.
}

VAL_ITER_IDS = {
    'gridEnv1': 'gridEnv1_val_iter_2021-05-14-15-54-10',
    'gridEnv4': 'gridEnv4_val_iter_2021-06-16-10-08-44',
    'multiPathsEnv': 'multiPathsEnv_val_iter_2021-06-04-19-31-25',
    'pendulum': 'pendulum_val_iter_2021-05-24-11-48-50',
    'mountaincar': 'mountaincar_val_iter_2021-09-15-18-56-32',
}

EPSILONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EXP_SETUP_1_IDS = {
    'gridEnv1': {
        'cov': [
            'gridEnv1_offline_dqn_2021-10-01-01-20-43.tar.gz',
            'gridEnv1_offline_dqn_2021-10-11-00-50-43.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-02-53-02.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-04-25-55.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-05-58-46.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-07-31-35.tar.gz',
        ],
        'no_cov': [
            'gridEnv1_offline_dqn_2021-10-01-09-04-33.tar.gz',
            'gridEnv1_offline_dqn_2021-10-11-01-46-50.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-10-37-23.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-12-10-29.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-13-43-44.tar.gz',
            'gridEnv1_offline_dqn_2021-10-01-15-16-33.tar.gz',
        ],
    },

    'gridEnv4': {
        'cov': [
            'gridEnv4_offline_dqn_2021-10-01-15-44-34.tar.gz',
            'gridEnv4_offline_dqn_2021-10-11-02-44-38.tar.gz',
            'gridEnv4_offline_dqn_2021-10-01-16-28-41.tar.gz',
            'gridEnv4_offline_dqn_2021-10-01-17-13-19.tar.gz',
            'gridEnv4_offline_dqn_2021-10-01-17-57-41.tar.gz',
            'gridEnv4_offline_dqn_2021-10-01-18-41-59.tar.gz',
        ],
        'no_cov': [
            'gridEnv4_offline_dqn_2021-10-09-16-09-35.tar.gz',
            'gridEnv4_offline_dqn_2021-10-11-03-40-59.tar.gz',
            'gridEnv4_offline_dqn_2021-10-09-17-48-34.tar.gz',
            'gridEnv4_offline_dqn_2021-10-17-16-17-05.tar.gz',
            'gridEnv4_offline_dqn_2021-10-17-17-12-55.tar.gz',
            'gridEnv4_offline_dqn_2021-10-17-18-08-48.tar.gz',
        ],
    },

    'multiPathsEnv': {
        'cov': [
            'multiPathsEnv_offline_dqn_2021-10-09-19-56-04.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-10-00-42-28.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-09-22-40-17.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-17-19-11-35.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-17-19-51-25.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-17-20-31-34.tar.gz',
        ],
        'no_cov': [
            'multiPathsEnv_offline_dqn_2021-10-09-21-01-07.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-09-23-45-07.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-09-21-51-29.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-17-21-11-45.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-17-21-51-35.tar.gz',
            'multiPathsEnv_offline_dqn_2021-10-17-22-31-17.tar.gz',
        ],
    },

    'pendulum': {
        'cov': [
            'pendulum_offline_dqn_2021-10-10-02-21-23.tar.gz',
            'pendulum_offline_dqn_2021-10-10-04-15-51.tar.gz',
            'pendulum_offline_dqn_2021-10-10-06-10-32.tar.gz',
            'pendulum_offline_dqn_2021-10-18-00-39-03.tar.gz',
            'pendulum_offline_dqn_2021-10-18-01-32-36.tar.gz',
            'pendulum_offline_dqn_2021-10-18-02-25-40.tar.gz',
        ],
        'no_cov': [
            'pendulum_offline_dqn_2021-10-10-03-18-38.tar.gz',
            'pendulum_offline_dqn_2021-10-10-05-13-42.tar.gz',
            'pendulum_offline_dqn_2021-10-10-07-08-12.tar.gz',
            'pendulum_offline_dqn_2021-10-18-03-18-48.tar.gz',
            'pendulum_offline_dqn_2021-10-18-04-11-32.tar.gz',
            'pendulum_offline_dqn_2021-10-18-05-04-33.tar.gz',
        ],
    },

}

OPTIMAL_SAMPLING_DIST_IDS = {
    'gridEnv1': [
        'gridEnv1_sampling_dist_2021-10-06-10-48-59', 'gridEnv1_sampling_dist_2021-10-06-10-49-04', 'gridEnv1_sampling_dist_2021-10-06-10-49-08', 'gridEnv1_sampling_dist_2021-10-06-10-49-12', 'gridEnv1_sampling_dist_2021-10-06-10-49-16', 'gridEnv1_sampling_dist_2021-10-06-10-49-21', 'gridEnv1_sampling_dist_2021-10-06-10-49-25', 'gridEnv1_sampling_dist_2021-10-06-10-49-29', 'gridEnv1_sampling_dist_2021-10-06-10-49-33', 'gridEnv1_sampling_dist_2021-10-06-10-49-37', 'gridEnv1_sampling_dist_2021-10-06-10-49-42', 'gridEnv1_sampling_dist_2021-10-06-10-49-46', 'gridEnv1_sampling_dist_2021-10-06-10-49-50', 'gridEnv1_sampling_dist_2021-10-06-10-49-54', 'gridEnv1_sampling_dist_2021-10-06-10-49-58', 'gridEnv1_sampling_dist_2021-10-06-10-50-03', 'gridEnv1_sampling_dist_2021-10-06-10-50-07', 'gridEnv1_sampling_dist_2021-10-06-10-50-11', 'gridEnv1_sampling_dist_2021-10-06-10-50-15', 'gridEnv1_sampling_dist_2021-10-06-10-50-19',
    ],
    'gridEnv4': [
        'gridEnv4_sampling_dist_2021-10-05-10-52-48', 'gridEnv4_sampling_dist_2021-10-05-10-52-50', 'gridEnv4_sampling_dist_2021-10-05-10-52-53', 'gridEnv4_sampling_dist_2021-10-05-10-52-56', 'gridEnv4_sampling_dist_2021-10-05-10-52-59', 'gridEnv4_sampling_dist_2021-10-05-10-53-01', 'gridEnv4_sampling_dist_2021-10-05-10-53-04', 'gridEnv4_sampling_dist_2021-10-05-10-53-07', 'gridEnv4_sampling_dist_2021-10-05-10-53-09', 'gridEnv4_sampling_dist_2021-10-05-10-53-12', 'gridEnv4_sampling_dist_2021-10-05-10-53-15', 'gridEnv4_sampling_dist_2021-10-05-10-53-18', 'gridEnv4_sampling_dist_2021-10-05-10-53-20', 'gridEnv4_sampling_dist_2021-10-05-10-53-23', 'gridEnv4_sampling_dist_2021-10-05-10-53-26', 'gridEnv4_sampling_dist_2021-10-05-10-53-28', 'gridEnv4_sampling_dist_2021-10-05-10-53-31', 'gridEnv4_sampling_dist_2021-10-05-10-53-34', 'gridEnv4_sampling_dist_2021-10-05-10-53-37', 'gridEnv4_sampling_dist_2021-10-05-10-53-39'
    ],
    'multiPathsEnv': [
        'multiPathsEnv_sampling_dist_2021-10-07-00-49-56', 'multiPathsEnv_sampling_dist_2021-10-07-00-49-58', 'multiPathsEnv_sampling_dist_2021-10-07-00-49-59', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-01', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-03', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-04', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-06', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-07', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-09', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-11', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-12', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-14', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-15', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-17', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-19', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-20', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-22', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-24', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-25', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-27',
    ],
    'pendulum': [
        'pendulum_sampling_dist_2021-10-07-18-40-46', 'pendulum_sampling_dist_2021-10-07-18-40-51', 'pendulum_sampling_dist_2021-10-07-18-40-57', 'pendulum_sampling_dist_2021-10-07-18-41-03', 'pendulum_sampling_dist_2021-10-07-18-41-09', 'pendulum_sampling_dist_2021-10-07-18-41-14', 'pendulum_sampling_dist_2021-10-07-18-41-20', 'pendulum_sampling_dist_2021-10-07-18-41-26', 'pendulum_sampling_dist_2021-10-07-18-41-32', 'pendulum_sampling_dist_2021-10-07-18-41-37', 'pendulum_sampling_dist_2021-10-07-18-41-43', 'pendulum_sampling_dist_2021-10-07-18-41-49', 'pendulum_sampling_dist_2021-10-07-18-41-55', 'pendulum_sampling_dist_2021-10-07-18-42-00', 'pendulum_sampling_dist_2021-10-07-18-42-06', 'pendulum_sampling_dist_2021-10-07-18-42-12', 'pendulum_sampling_dist_2021-10-07-18-42-18', 'pendulum_sampling_dist_2021-10-07-18-42-23', 'pendulum_sampling_dist_2021-10-07-18-42-29', 'pendulum_sampling_dist_2021-10-07-18-42-35'
    ],
}

#################################################################


FIGURE_X = 6.0
FIGURE_Y = 4.0

PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

RED_COLOR = (0.886, 0.29, 0.20)
GRAY_COLOR = (0.2,0.2,0.2)
BLUE_COLOR = (58/255,138/255,189/255)

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

def f_div(x, y):
        y = y + 1e-06
        # return np.dot(y, ((x/y)-1)**2 )
        return np.dot(y, (x/y)**2 - 1)

def main():

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + '/appendix_plots'
    os.makedirs(output_folder, exist_ok=True)

    # Load optimal policy/Q-values.
    val_iter_data = {}
    for (env_id, val_iter_id) in VAL_ITER_IDS.items():
        val_iter_path = DATA_FOLDER_PATH + val_iter_id
        print(f"Opening experiment {val_iter_id}")
        with open(val_iter_path + "/train_data.json", 'r') as f:
            d = json.load(f)
            d = json.loads(d)
            d = d[0]
        f.close()
        val_iter_data[env_id] = {'Q_vals': np.array(d['Q_vals'])} # [S,A]

    # Load data.
    data = {}
    for cov_type, cov_type_data in EXP_SETUP_1_IDS[ENV_ID_TO_PLOT].items():

        env_data = []
        for exp_id in cov_type_data:

            print(f"Opening experiment {exp_id}")

            exp_path = DATA_FOLDER_PATH + exp_id
            exp_name = pathlib.Path(exp_path).stem
            exp_name = '.'.join(exp_name.split('.')[:-1])

            tar = tarfile.open(exp_path)
            data_file = tar.extractfile("{0}/train_data.json".format(exp_name))

            exp_data = json.load(data_file)
            exp_data = json.loads(exp_data)

            # Parse data.
            parsed_data = {}
            parsed_data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]
            parsed_data['replay_buffer_counts'] = np.array([e['replay_buffer_counts'] for e in exp_data]) # [R,(S),S,A]
            parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards']
                            for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]

            env_data.append(parsed_data)

        data[cov_type] = env_data

    # Q-values error.
    """ width = 0.35
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for cov_type, cov_type_data in data.items():

        if cov_type == 'cov':
            cov_offset = width/2
            color = BLUE_COLOR
            coverage_text = 'True'
        else:
            color = RED_COLOR
            cov_offset = - width/2
            coverage_text = 'False'

        for x, exp_data in enumerate(cov_type_data):

            errors = np.abs(val_iter_data[ENV_ID_TO_PLOT]['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
            errors = np.mean(errors, axis=(2,3)) # [R,E]
            errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)
            point_est, (lower_ci, upper_ci) = mean_agg_func(errors)
            lower_ci = np.abs(lower_ci - point_est)
            upper_ci = np.abs(upper_ci - point_est)

            if x == 0:
                plt.bar(x + cov_offset, point_est, width,
                yerr=[[lower_ci],[upper_ci]], color=color, label=f'Coverage={coverage_text}')
            else:
                plt.bar(x + cov_offset, point_est, width,
                yerr=[[lower_ci],[upper_ci]], color=color)
    
    plt.xticks(np.arange(len(EPSILONS)), labels=EPSILONS)
    plt.ylabel(r'$Q$-values error')
    plt.xlabel(r'$\epsilon$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_qvals_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_qvals_error.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0) """

    if ENV_ID_TO_PLOT in env_suite.CUSTOM_GRID_ENVS.keys():
        rollouts_types = sorted(env_suite.CUSTOM_GRID_ENVS[ENV_ID_TO_PLOT].keys())
    elif ENV_ID_TO_PLOT == 'pendulum':
        rollouts_types = sorted(env_suite.PENDULUM_ENVS.keys())
    elif ENV_ID_TO_PLOT == 'mountaincar':
        rollouts_types = sorted(env_suite.MOUNTAINCAR_ENVS.keys())
    elif ENV_ID_TO_PLOT == 'multiPathsEnv':
        rollouts_types = sorted(env_suite.MULTIPATHS_ENVS.keys())
    else:
        raise ValueError(f'Env. {ENV_ID_TO_PLOT} does not have rollout types defined.')

    width = 0.35
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for cov_type, cov_type_data in data.items():

        if cov_type == 'cov':
            cov_offset = width/2
            color = BLUE_COLOR
            coverage_text = 'True'
        else:
            color = RED_COLOR
            cov_offset = - width/2
            coverage_text = 'False'

        for x, exp_data in enumerate(cov_type_data):

            default_rollout_idx = rollouts_types.index('default')
            rewards = exp_data['rollouts_rewards'][:,:,default_rollout_idx,:] # [R,(E),num_rollouts]
            rewards = np.mean(rewards[:,-10:,:], axis=(1,2)) # [R]
            print(rewards)

            point_est, (lower_ci, upper_ci) = mean_agg_func(rewards)
            lower_ci = np.abs(lower_ci - point_est)
            upper_ci = np.abs(upper_ci - point_est)

            if x == 0:
                plt.bar(x + cov_offset, point_est, width,
                yerr=[[lower_ci],[upper_ci]], color=color, label=f'Coverage={coverage_text}')
            else:
                plt.bar(x + cov_offset, point_est, width,
                yerr=[[lower_ci],[upper_ci]], color=color)
    
    plt.xticks(np.arange(len(EPSILONS)), labels=EPSILONS)
    plt.ylabel('Reward')
    plt.xlabel(r'$\epsilon$')
    plt.legend()
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_reward.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_reward.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Coverage plots.
    """ width = 0.35

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for cov_type, cov_type_data in data.items():

        if cov_type == 'cov':
            cov_offset = width/2
            color = BLUE_COLOR
            coverage_text = 'True'
        else:
            color = RED_COLOR
            cov_offset = - width/2
            coverage_text = 'False'


        for x, exp_data in enumerate(cov_type_data):

            sa_counts = exp_data['replay_buffer_counts'] # [R,E,S,A]
            sa_counts = sa_counts[:,-1,:,:] # [R,S,A]
            sa_counts = np.mean(sa_counts,axis=0) # [S,A]
            coverage = np.sum(sa_counts > 0) / (sa_counts.shape[0]*sa_counts.shape[1]) # + (1.0 - 0.78125)
            print('coverage', coverage)

            if x == 0:
                plt.bar(x + cov_offset, coverage, width, color=color, label=f'Coverage={coverage_text}')
            else:
                plt.bar(x + cov_offset, coverage, width, color=color)
    
    plt.xticks(np.arange(len(EPSILONS)), labels=EPSILONS)
    plt.ylabel('Coverage (\%)')
    plt.xlabel(r'$\epsilon$')
    plt.ylim(0.0, 1.0)
    plt.legend(loc=4)
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_coverage.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_coverage.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0) """

    # Distance plots.
    """ optimal_dists = []
    for optimal_dist_id in OPTIMAL_SAMPLING_DIST_IDS[ENV_ID_TO_PLOT]:
        optimal_dist_path = DATA_FOLDER_PATH + optimal_dist_id + '/data.json'
        print(optimal_dist_path)
        with open(optimal_dist_path, 'r') as f:
            d = json.load(f)
            d = np.array(json.loads(d)['sampling_dist'])
        f.close()
        optimal_dists.append(d)
    print('len(optimal_dists)', len(optimal_dists))

    width = 0.35
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for cov_type, cov_type_data in data.items():

        if cov_type == 'cov':
            cov_offset = width/2
            color = BLUE_COLOR
            coverage_text = 'True'
        else:
            color = RED_COLOR
            cov_offset = - width/2
            coverage_text = 'False'

        for x, exp_data in enumerate(cov_type_data):

            sa_counts = exp_data['replay_buffer_counts'] # [R,E,S,A]
            sa_counts = sa_counts[:,-1,:,:] # [R,S,A]
            sa_counts = np.mean(sa_counts,axis=0) # [S,A]
            sa_dist = sa_counts / np.sum(sa_counts)
            sa_dist = sa_dist.flatten()

            div = np.min([f_div(optimal_dist,sa_dist)
                            for optimal_dist in optimal_dists])
            if x == 0:
                plt.bar(x + cov_offset, div, width, color=color, label=f'Coverage={coverage_text}')
            else:
                plt.bar(x + cov_offset, div, width, color=color)
    
    plt.xticks(np.arange(len(EPSILONS)), labels=EPSILONS)
    plt.ylabel(r'$\min_{\pi^*}\chi^2(d_{\pi^*}||\mu)$')
    plt.xlabel(r'$\epsilon$')
    plt.legend(loc=4)
    plt.yscale('log')
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_distance.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{ENV_ID_TO_PLOT}_distance.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0) """

if __name__ == '__main__':
    main()
