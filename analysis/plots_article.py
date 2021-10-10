import os
import json
import tarfile
import numpy as np 
import scipy
from scipy.spatial import distance
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('ggplot')

from envs import env_suite

#################################################################
EXP_SETUP_TO_PLOT = 2 # [1, 2, 3, or 4]
RENDER_LATEX = False
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

EXP_SETUP_1_IDS = {
    'gridEnv1': [
        'gridEnv1_offline_dqn_2021-09-29-13-01-33.tar.gz',
        'gridEnv1_offline_dqn_2021-09-29-14-08-27.tar.gz',
        'gridEnv1_offline_dqn_2021-09-29-15-15-14.tar.gz',
        'gridEnv1_offline_dqn_2021-09-29-16-22-31.tar.gz',
        'gridEnv1_offline_dqn_2021-09-29-17-29-30.tar.gz',
        'gridEnv1_offline_dqn_2021-09-29-18-36-18.tar.gz',
        'gridEnv1_offline_dqn_2021-09-29-19-42-46.tar.gz',
    ],
    'gridEnv4': [
        'gridEnv4_offline_dqn_2021-09-29-20-48-59.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-21-55-21.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-23-02-01.tar.gz',
        'gridEnv4_offline_dqn_2021-09-30-00-08-35.tar.gz',
        'gridEnv4_offline_dqn_2021-09-30-01-15-01.tar.gz',
        'gridEnv4_offline_dqn_2021-09-30-02-20-59.tar.gz',
        'gridEnv4_offline_dqn_2021-09-30-03-27-01.tar.gz',
    ],
    'multiPathsEnv': [
        'multiPathsEnv_offline_dqn_2021-09-30-04-32-55.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-30-05-05-06.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-30-05-37-24.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-30-06-09-36.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-30-06-42-02.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-30-07-13-56.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-30-07-46-20.tar.gz',
    ],
    'pendulum': [
        'pendulum_offline_dqn_2021-09-30-08-18-18.tar.gz',
        'pendulum_offline_dqn_2021-09-30-09-34-33.tar.gz',
        'pendulum_offline_dqn_2021-09-30-10-51-08.tar.gz',
        'pendulum_offline_dqn_2021-09-30-12-07-27.tar.gz',
        'pendulum_offline_dqn_2021-09-30-13-23-51.tar.gz',
        'pendulum_offline_dqn_2021-09-30-14-51-05.tar.gz',
        'pendulum_offline_dqn_2021-09-30-16-07-30.tar.gz',
    ],
    'mountaincar': [
        'mountaincar_offline_dqn_2021-09-30-17-23-26.tar.gz',
        'mountaincar_offline_dqn_2021-09-30-19-13-16.tar.gz',
        'mountaincar_offline_dqn_2021-09-30-21-02-34.tar.gz',
        'mountaincar_offline_dqn_2021-09-30-22-51-20.tar.gz',
        'mountaincar_offline_dqn_2021-10-01-00-40-35.tar.gz',
        'mountaincar_offline_dqn_2021-10-01-02-35-59.tar.gz',
        'mountaincar_offline_dqn_2021-10-01-04-31-09.tar.gz',
    ],
}

EXP_SETUP_2_IDS = {
    'gridEnv1': [
        'gridEnv1_offline_dqn_2021-09-28-10-21-36.tar.gz',
        'gridEnv1_offline_dqn_2021-09-28-12-11-46.tar.gz',
        'gridEnv1_offline_dqn_2021-09-28-13-45-48.tar.gz',
        'gridEnv1_offline_dqn_2021-09-28-15-20-03.tar.gz',
        'gridEnv1_offline_dqn_2021-09-28-16-54-14.tar.gz',
        'gridEnv1_offline_dqn_2021-09-28-18-27-46.tar.gz',
    ],
    'gridEnv4': [
        'gridEnv4_offline_dqn_2021-09-29-01-05-19.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-02-38-34.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-04-11-55.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-05-45-21.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-07-18-35.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-08-51-52.tar.gz',
        'gridEnv4_offline_dqn_2021-09-29-10-24-54.tar.gz',
    ],
    'multiPathsEnv': [
        'multiPathsEnv_offline_dqn_2021-09-29-14-27-54.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-29-15-21-02.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-29-16-14-11.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-29-17-04-42.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-29-17-52-40.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-29-18-40-20.tar.gz',
        'multiPathsEnv_offline_dqn_2021-09-29-19-28-22.tar.gz',
    ],
    'pendulum': [
        'pendulum_offline_dqn_2021-10-08-11-35-33.tar.gz',
        'pendulum_offline_dqn_2021-10-10-20-52-08.tar.gz',
        'pendulum_offline_dqn_2021-10-10-22-27-30.tar.gz',
        'pendulum_offline_dqn_2021-10-08-13-50-42.tar.gz',
        'pendulum_offline_dqn_2021-10-08-14-35-03.tar.gz',
        'pendulum_offline_dqn_2021-10-08-19-19-39.tar.gz',
        'pendulum_offline_dqn_2021-10-08-20-15-25.tar.gz',
    ],
    'mountaincar': [
        'mountaincar_offline_dqn_2021-10-09-02-16-14.tar.gz',
        'mountaincar_offline_dqn_2021-10-09-03-28-52.tar.gz',
        'mountaincar_offline_dqn_2021-10-09-04-40-14.tar.gz',
        'mountaincar_offline_dqn_2021-10-09-05-58-17.tar.gz',
        'mountaincar_offline_dqn_2021-10-09-07-20-37.tar.gz',
        'mountaincar_offline_dqn_2021-10-09-08-36-16.tar.gz',
        'mountaincar_offline_dqn_2021-10-09-09-55-54.tar.gz',
    ],
}
#################################################################
if RENDER_LATEX:
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

    print(f'Creating plots for experimental setup {EXP_SETUP_TO_PLOT}.')

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + '/article_plots'
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

    if EXP_SETUP_TO_PLOT in [1,2]:
        """
            Plots for experimental setups 1 and 2.
        """
        if EXP_SETUP_TO_PLOT == 1:
            exp_setup_ids = EXP_SETUP_1_IDS
        else:
            exp_setup_ids = EXP_SETUP_2_IDS

        # Load data.
        data = {}
        for env_id, env_exp_ids in exp_setup_ids.items():

            env_data = {}
            for exp_id in env_exp_ids:

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
                parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards']
                                for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]
                parsed_data['replay_buffer_counts'] = np.array([e['replay_buffer_counts']
                                for e in exp_data]) # [R,(E),S,A]

                env_data[exp_id] = parsed_data

            data[env_id] = env_data


        # X: entropy, Y: Q-values average error (w/ conf. interval).
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for (env_id, env_data) in data.items():

            X = []
            Y = []
            lower_ci_bounds, upper_ci_bounds = [], []
            for (exp_id, exp_data) in env_data.items():
                errors = np.abs(val_iter_data[env_id]['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
                errors = np.mean(errors, axis=(2,3)) # [R,E]
                errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)

                point_est, (lower_ci, upper_ci) = mean_agg_func(errors)
                
                entropies = []
                for _, run_data in enumerate(exp_data['replay_buffer_counts']): # run_data = [(E),S,A]
                    run_data = np.mean(run_data[-10:,:,:],axis=0) # [S,A] (use last 10 episodes data)
                    total_sum = np.sum(run_data) # []
                    sa_probs = run_data.flatten() / (total_sum + 1e-08) # [(E),S,A]
                    entropy = -np.dot(sa_probs, np.log(sa_probs + 1e-08)) # []
                    entropies.append(entropy)
                dist_entropy = np.mean(entropies)

                X.append(dist_entropy)
                Y.append(point_est)
                lower_ci_bounds.append(lower_ci)
                upper_ci_bounds.append(upper_ci)

            p = plt.plot(X, Y, label=ENVS_LABELS[env_id])
            plt.fill_between(X, lower_ci_bounds, upper_ci_bounds,
                        color=p[0].get_color(), alpha=0.15)

        if RENDER_LATEX:
            plt.ylabel(r'$Q$-values error')
            plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')
        else:
            plt.ylabel('$Q$-values error')
            plt.xlabel('$\mathbb{E}[\mathcal{H}(\mu)]$')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_entropy_QvalsError.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_entropy_QvalsError.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)


        # X: normalized entropy, Y: Q-values average error (w/ conf. interval).
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for (env_id, env_data) in data.items():

            X = []
            Y = []
            lower_ci_bounds, upper_ci_bounds = [], []
            for (exp_id, exp_data) in env_data.items():
                errors = np.abs(val_iter_data[env_id]['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
                errors = np.mean(errors, axis=(2,3)) # [R,E]
                errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)

                point_est, (lower_ci, upper_ci) = mean_agg_func(errors)

                entropies = []
                for _, run_data in enumerate(exp_data['replay_buffer_counts']): # run_data = [(E),S,A]
                    run_data = np.mean(run_data[-10:,:,:],axis=0) # [S,A] (use last 10 episodes data)
                    total_sum = np.sum(run_data) # []
                    sa_probs = run_data.flatten() / (total_sum + 1e-08) # [(E),S,A]
                    entropy = -np.dot(sa_probs, np.log(sa_probs + 1e-08)) # []
                    entropies.append(entropy)
                dist_entropy = np.mean(entropies)

                X.append(dist_entropy / UNIFORM_SAMPLING_DIST_ENTROPY[env_id])
                Y.append(point_est)
                lower_ci_bounds.append(lower_ci)
                upper_ci_bounds.append(upper_ci)

            p = plt.plot(X, Y, label=ENVS_LABELS[env_id])
            plt.fill_between(X, lower_ci_bounds, upper_ci_bounds,
                        color=p[0].get_color(), alpha=0.15)

        if RENDER_LATEX:
            plt.ylabel(r'$Q$-values error')
            plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)] / \mathcal{H}(\mathcal{U})$')
        else:
            plt.ylabel('$Q$-values error')
            plt.xlabel('$\mathbb{E}[\mathcal{H}(\mu)] / \mathcal{H}(\mathcal{U})$')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_normEntropy_QvalsError.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_normEntropy_QvalsError.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)


        # X: normalized entropy, Y: default reward (w/ conf. interval).
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for (env_id, env_data) in data.items():

            if env_id in env_suite.CUSTOM_GRID_ENVS.keys():
                rollouts_types = sorted(env_suite.CUSTOM_GRID_ENVS[env_id].keys())
            elif env_id == 'pendulum':
                rollouts_types = sorted(env_suite.PENDULUM_ENVS.keys())
            elif env_id == 'mountaincar':
                rollouts_types = sorted(env_suite.MOUNTAINCAR_ENVS.keys())
            elif env_id == 'multiPathsEnv':
                rollouts_types = sorted(env_suite.MULTIPATHS_ENVS.keys())
            else:
                raise ValueError(f'Env. {env_id} does not have rollout types defined.')

            X = []
            Y = []
            lower_ci_bounds, upper_ci_bounds = [], []
            for (exp_id, exp_data) in env_data.items():

                default_rollout_idx = rollouts_types.index('default')
                rewards = exp_data['rollouts_rewards'][:,:,default_rollout_idx,:] # [R,(E),num_rollouts]
                rewards = np.mean(rewards[:,-10:,:], axis=(1,2)) # [R]

                point_est, (lower_ci, upper_ci) = mean_agg_func(rewards)

                entropies = []
                for _, run_data in enumerate(exp_data['replay_buffer_counts']): # run_data = [(E),S,A]
                    run_data = np.mean(run_data[-10:,:,:],axis=0) # [S,A] (use last 10 episodes data)
                    total_sum = np.sum(run_data) # []
                    sa_probs = run_data.flatten() / (total_sum + 1e-08) # [(E),S,A]
                    entropy = -np.dot(sa_probs, np.log(sa_probs + 1e-08)) # []
                    entropies.append(entropy)
                dist_entropy = np.mean(entropies)

                X.append(dist_entropy / UNIFORM_SAMPLING_DIST_ENTROPY[env_id])
                Y.append(point_est)
                lower_ci_bounds.append(lower_ci)
                upper_ci_bounds.append(upper_ci)

            p = plt.plot(X, Y, label=ENVS_LABELS[env_id])
            plt.fill_between(X, lower_ci_bounds, upper_ci_bounds,
                        color=p[0].get_color(), alpha=0.15)

        plt.ylabel('Reward')
        if RENDER_LATEX:
            plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)] / \mathcal{H}(\mathcal{U})$')
        else:
            plt.xlabel('$\mathbb{E}[\mathcal{H}(\mu)] / \mathcal{H}(\mathcal{U})$')
        plt.yscale('linear')
        plt.legend()
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_normEntropy_reward.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_normEntropy_reward.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)


        # X: normalized entropy, Y: normalized default reward (w/ conf. interval).
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for (env_id, env_data) in data.items():

            if env_id in env_suite.CUSTOM_GRID_ENVS.keys():
                rollouts_types = sorted(env_suite.CUSTOM_GRID_ENVS[env_id].keys())
            elif env_id == 'pendulum':
                rollouts_types = sorted(env_suite.PENDULUM_ENVS.keys())
            elif env_id == 'mountaincar':
                rollouts_types = sorted(env_suite.MOUNTAINCAR_ENVS.keys())
            elif env_id == 'multiPathsEnv':
                rollouts_types = sorted(env_suite.MULTIPATHS_ENVS.keys())
            else:
                raise ValueError(f'Env. {env_id} does not have rollout types defined.')

            X = []
            Y = []
            lower_ci_bounds, upper_ci_bounds = [], []
            for (exp_id, exp_data) in env_data.items():

                default_rollout_idx = rollouts_types.index('default')
                rewards = exp_data['rollouts_rewards'][:,:,default_rollout_idx,:] # [R,(E),num_rollouts]
                rewards = np.mean(rewards[:,-10:,:], axis=(1,2)) # [R]
                rewards = rewards / MAXIMUM_REWARD[env_id]

                point_est, (lower_ci, upper_ci) = mean_agg_func(rewards)

                entropies = []
                for _, run_data in enumerate(exp_data['replay_buffer_counts']): # run_data = [(E),S,A]
                    run_data = np.mean(run_data[-10:,:,:],axis=0) # [S,A] (use last 10 episodes data)
                    total_sum = np.sum(run_data) # []
                    sa_probs = run_data.flatten() / (total_sum + 1e-08) # [(E),S,A]
                    entropy = -np.dot(sa_probs, np.log(sa_probs + 1e-08)) # []
                    entropies.append(entropy)
                dist_entropy = np.mean(entropies)

                X.append(dist_entropy / UNIFORM_SAMPLING_DIST_ENTROPY[env_id])
                Y.append(point_est)
                lower_ci_bounds.append(lower_ci)
                upper_ci_bounds.append(upper_ci)

            p = plt.plot(X, Y, label=ENVS_LABELS[env_id])
            plt.fill_between(X, lower_ci_bounds, upper_ci_bounds,
                        color=p[0].get_color(), alpha=0.15)

        plt.ylabel('Normalized reward')
        if RENDER_LATEX:
            plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)] / \mathcal{H}(\mathcal{U})$')
        else:
            plt.xlabel('$\mathbb{E}[\mathcal{H}(\mu)] / \mathcal{H}(\mathcal{U})$')
        plt.yscale('linear')
        plt.legend()
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_normEntropy_normReward.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_folder}/expSetup{EXP_SETUP_TO_PLOT}_normEntropy_normReward.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    elif EXP_SETUP_TO_PLOT == 3:
        raise ValueError('Not implemented.')

    elif EXP_SETUP_TO_PLOT == 4:
        raise ValueError('Not implemented.')

    else:
        raise ValueError('Unknown experimental setup.')


if __name__ == '__main__':
    main()
