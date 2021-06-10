import os
import sys
import math
import json
import random
import numpy as np
import pathlib
import scipy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import statsmodels.api as sm

FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/mdp_1/'

VAL_ITER_DATA = 'customEnv1_val_iter_2021-06-09-11-40-56'

DATA_1 = ['customEnv1_linear_approximator_2021-06-09-12-07-49', 'customEnv1_linear_approximator_2021-06-09-12-08-56', 'customEnv1_linear_approximator_2021-06-09-12-10-03', 'customEnv1_linear_approximator_2021-06-09-12-11-11', 'customEnv1_linear_approximator_2021-06-09-12-12-18', 'customEnv1_linear_approximator_2021-06-09-12-13-24', 'customEnv1_linear_approximator_2021-06-09-12-14-31', 'customEnv1_linear_approximator_2021-06-09-12-15-38', 'customEnv1_linear_approximator_2021-06-09-12-16-45', 'customEnv1_linear_approximator_2021-06-09-12-17-53']
DATA_2 = ['customEnv1_linear_approximator_2021-06-09-12-26-21', 'customEnv1_linear_approximator_2021-06-09-12-27-29', 'customEnv1_linear_approximator_2021-06-09-12-28-37', 'customEnv1_linear_approximator_2021-06-09-12-29-45', 'customEnv1_linear_approximator_2021-06-09-12-30-53', 'customEnv1_linear_approximator_2021-06-09-12-32-01', 'customEnv1_linear_approximator_2021-06-09-12-33-09', 'customEnv1_linear_approximator_2021-06-09-12-34-17', 'customEnv1_linear_approximator_2021-06-09-12-35-25']
DATA_3 = ['customEnv1_linear_approximator_2021-06-09-13-30-31', 'customEnv1_linear_approximator_2021-06-09-13-31-38', 'customEnv1_linear_approximator_2021-06-09-13-32-45', 'customEnv1_linear_approximator_2021-06-09-13-33-51']
DATA = DATA_1 + DATA_2 + DATA_3
ALPHAS_1 = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
ALPHAS_2 = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
ALPHAS_3 = [250.0, 500.0, 750.0, 1000.0]
ALPHAS = ALPHAS_1 + ALPHAS_2 + ALPHAS_3

if __name__ == "__main__":

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Entropy plot.
    entropies = []
    for alpha in ALPHAS:
        expected_entropy = scipy.special.digamma((4*2)*alpha + 1) - scipy.special.digamma(alpha + 1)
        entropies.append(expected_entropy)

    # Load optimal policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + VAL_ITER_DATA
    print(f"Opening experiment {VAL_ITER_DATA}")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        val_iter_data = json.load(f)
        val_iter_data = json.loads(val_iter_data)
        val_iter_data = val_iter_data[0]
    f.close()
    val_iter_data['Q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]

    # Load and parse data.
    data = {}
    for exp in DATA:

        # Open data.
        print(f"Opening experiment {exp}")
        exp_path = DATA_FOLDER_PATH + exp
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            exp_data = json.loads(exp_data)
        f.close()

        # Parse data for each train run.
        parsed_data = {}

        # episode_rewards field.
        parsed_data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data]) # [R,E]

        # Q_vals field.
        parsed_data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]

        data[exp] = parsed_data

    """
        Episode rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    to_plot = []
    for exp in DATA:
        rewards = data[exp]['episode_rewards'] # [R,E]
        rewards = rewards[:,-500:] # Filter.
        to_plot.append(np.mean(rewards)) # []

    plt.plot(ALPHAS, to_plot)

    plt.xlabel('Alpha')
    plt.ylabel('Reward')

    plt.savefig('{0}/episode_rewards_avg.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards_avg.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    to_plot = []
    for exp in DATA:
        rewards = data[exp]['episode_rewards'] # [R,E]
        rewards = rewards[:,-500:] # Filter.
        to_plot.append(np.mean(rewards)) # []

    plt.plot(entropies, to_plot)

    plt.xlabel('Expected entropy')
    plt.ylabel('Reward')

    plt.savefig('{0}/episode_rewards_avg_2.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards_avg_2.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Q-values plots.
    """
    # Sum plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    to_plot = []
    for exp in DATA:
        errors = np.abs(val_iter_data['Q_vals'] - data[exp]['Q_vals']) # [R,E,S,A]
        errors = errors[:,-500:,:,:]
        errors = np.mean(errors, axis=(0,1)) # [S,A]
        to_plot.append(np.sum(errors))

    plt.plot(ALPHAS, to_plot)

    plt.xlabel('Alpha')
    plt.ylabel('Summed Q-values error')

    plt.savefig('{0}/q_values_summed_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_summed_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Sum plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    to_plot = []
    for exp in DATA:
        errors = np.abs(val_iter_data['Q_vals'] - data[exp]['Q_vals']) # [R,E,S,A]
        errors = errors[:,-500:,:,:]
        errors = np.mean(errors, axis=(0,1)) # [S,A]
        to_plot.append(np.sum(errors))

    plt.plot(entropies, to_plot)

    plt.xlabel('Expected entropy')
    plt.ylabel('Summed Q-values error')

    plt.savefig('{0}/q_values_summed_error_2.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_summed_error_2.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Max plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    to_plot = []
    for exp in DATA:
        errors = np.abs(val_iter_data['Q_vals'] - data[exp]['Q_vals']) # [R,E,S,A]
        errors = errors[:,-500:,:,:]
        errors = np.mean(errors, axis=(0,1)) # [S,A]
        to_plot.append(np.max(errors))

    plt.plot(ALPHAS, to_plot)

    plt.xlabel('Alpha')
    plt.ylabel('Max Q-values error')

    plt.savefig('{0}/q_values_max_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_max_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    to_plot = []
    for exp in DATA:
        errors = np.abs(val_iter_data['Q_vals'] - data[exp]['Q_vals']) # [R,E,S,A]
        errors = errors[:,-500:,:,:]
        errors = np.mean(errors, axis=(0,1)) # [S,A]
        to_plot.append(np.max(errors))

    plt.plot(entropies, to_plot)

    plt.xlabel('Expected entropy')
    plt.ylabel('Max Q-values error')

    plt.savefig('{0}/q_values_max_error_2.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_max_error_2.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()


    """
        Replay buffer entropy.
    """
    # fig = plt.figure()
    # fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # for i, run_data in enumerate(data['replay_buffer_counts']): # run_data = [(E),S,A]

    #     aggregated_data = np.sum(run_data, axis=2) # [(E),S]
    #     row_sums = np.sum(aggregated_data, axis=1) # [(E)]
    #     state_probs = aggregated_data / (row_sums[:, np.newaxis] + 1e-05) # [(E), S]

    #     Y = -np.sum(state_probs * np.log(state_probs+1e-8), axis=1)
    #     X = data['replay_buffer_counts_episodes'] 
    #     plt.plot(X, Y, label=f'Run {i}')

    # plt.ylabel('H[P(s)]')
    # plt.xlabel('Episode')

    # plt.legend()

    # plt.title(f'Replay buffer: H[P(s)]')

    # plt.savefig(f'{replay_buffer_plots_path}/P_s_entropy.png', bbox_inches='tight', pad_inches=0)
    # plt.close()