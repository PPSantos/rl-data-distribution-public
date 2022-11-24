import os
import sys
import math
import json
import random
import numpy as np
import pathlib

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 12})

FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/output_plots/'

VAL_ITER_DATA = 'four_state_mdp_val_iter_2022-11-24-11-31-49' # gamma = 1.0
# DATA = [
#     'four_state_mdp_linear_approximator_2022-11-24-14-40-58',
#     'four_state_mdp_linear_approximator_2022-11-24-14-58-55',
#     'four_state_mdp_linear_approximator_2022-11-24-15-18-14',
# ]
# LABELS = [r'$\epsilon = 0.05$', r'$\epsilon = 1.0$', 'Uniform']
# KEYS = ['eps_05', 'eps_1', 'unif']

DATA = [
    'four_state_mdp_linear_approximator_2022-11-24-14-40-58',
    #'four_state_mdp_linear_approximator_2022-11-24-15-39-16',
    'four_state_mdp_linear_approximator_2022-11-24-15-52-37',
]
LABELS = [r'Replay buffer size = $\infty$', 'Replay buffer size = 10 000'] #, 'Replay buffer size = 50 000']
KEYS = ['replay_infty', 'replay_10_000'] # , 'replay_50_000']

if __name__ == "__main__":

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Load optimal policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + VAL_ITER_DATA
    print(f"Opening experiment {VAL_ITER_DATA}")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        val_iter_data = json.load(f)
        val_iter_data = json.loads(val_iter_data)
    f.close()
    val_iter_data['Q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]
    print(val_iter_data)

    # Load and parse data.
    data = []

    for d in DATA:

        # Open data.
        print(f"Opening experiment {d}")
        exp_path = DATA_FOLDER_PATH + d
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            exp_data = json.loads(exp_data)
        f.close()

        parsed_data = {}

        # episode_rewards field.
        parsed_data['episode_rewards'] = np.array([e['episode_rewards'] for e in exp_data]) # [R,E]

        # Q_vals field.
        parsed_data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]

        # weights field.
        parsed_data['weights'] = np.array([e['weights'] for e in exp_data]) # [R,E,F]

        # replay_buffer_counts field.
        parsed_data['replay_buffer_counts'] = np.array([e['replay_buffer_counts'] for e in exp_data]) # [R,E,S,A]

        # rollouts_rewards field.
        parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]

        data.append(parsed_data)

    # Load additional variables from last experiment file.
    Q_vals_episodes = exp_data[0]['Q_vals_episodes'] # [(E)]
    replay_buffer_counts_episodes = exp_data[0]['replay_buffer_counts_episodes'] # [(E)]
    rollouts_episodes = exp_data[0]['rollouts_episodes'] # [(E)]

    """
        Episode rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (d, lbl) in zip(data, LABELS):
        plt.plot(np.mean(d['episode_rewards'], axis=0), label=lbl)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.savefig('{0}/episode_rewards.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Rollouts rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (d, lbl) in zip(data, LABELS):
        rollouts_data = d['rollouts_rewards'][:,:,:] # [R,(E),num_rollouts]
        print(rollouts_data.shape)
        rollouts_data = np.mean(rollouts_data, axis=(0,2))
        rollouts_data = 100 * rollouts_data
        plt.plot(rollouts_episodes, rollouts_data, label=lbl)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim([-50, 110])
    plt.legend()

    plt.savefig('{0}/rollouts_episode_rewards.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/rollouts_episode_rewards.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Q-values plots.
    """
    for (d, k) in zip(data, KEYS):

        # Average error plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        q_vals_averaged = np.mean(d['Q_vals'], axis=0) # [E,S,A]
        plt.plot(Q_vals_episodes, 100*q_vals_averaged[:,0,0], label=r'$Q_w(s_1,a_1)$')
        plt.plot(Q_vals_episodes, 100*q_vals_averaged[:,0,1], label=r'$Q_w(s_1,a_2)$')
        plt.plot(Q_vals_episodes, 100*q_vals_averaged[:,1,0], label=r'$Q_w(s_2,a_1)$')
        plt.plot(Q_vals_episodes, 100*q_vals_averaged[:,1,1], label=r'$Q_w(s_2,a_2)$')

        plt.xlabel('Episode')
        plt.ylabel(r'$Q$-value')
        #plt.ylim([-0.02, 0.32])
        plt.legend()

        plt.savefig('{0}/{1}_q_values.pdf'.format(PLOTS_FOLDER_PATH, k), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/{1}_q_values.png'.format(PLOTS_FOLDER_PATH, k), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Average error plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (d, lbl) in zip(data, LABELS):
        errors = np.abs(val_iter_data['Q_vals'] - d['Q_vals']) # [R,E,S,A]
        errors = np.mean(errors, axis=(2,3)) # [R,E]
        errors = np.mean(errors, axis=0) # [E]
        plt.plot(Q_vals_episodes, errors, label=lbl)

    plt.xlabel('Episode')
    plt.ylabel(r'Mean $Q$-values error')
    plt.ylim([-0.02, 0.32])
    plt.legend(loc=4)

    plt.savefig('{0}/q_values_mean_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/q_values_mean_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Weights.
    """
    for (d, k) in zip(data, KEYS):
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        plt.plot(100*np.mean(d['weights'], axis=0)[:,0], label='$w_1$')
        plt.plot(100*np.mean(d['weights'], axis=0)[:,1], label='$w_2$')
        plt.plot(100*np.mean(d['weights'], axis=0)[:,2], label='$w_3$')

        plt.xlabel('Episode')
        plt.ylabel('Weight value')
        plt.legend()

        plt.savefig('{0}/{1}_weights.pdf'.format(PLOTS_FOLDER_PATH, k), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/{1}_weights.png'.format(PLOTS_FOLDER_PATH, k), bbox_inches='tight', pad_inches=0)
        plt.close()

    """
        Replay buffer plots.
    """
    for (d, k) in zip(data, KEYS):
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        run_data = np.mean(d['replay_buffer_counts'], axis=0) # [E,S,A]

        counts_0 = run_data[:,0,0] / np.sum(run_data, axis=(1,2))
        #counts_1 = run_data[:,0,1] / np.sum(run_data, axis=(1,2))
        counts_2 = run_data[:,1,0] / np.sum(run_data, axis=(1,2))
        #counts_3 = run_data[:,1,1] / np.sum(run_data, axis=(1,2))

        plt.plot(replay_buffer_counts_episodes, counts_0, label='$\mu(s_1, a_1$)')
        #plt.plot(replay_buffer_counts_episodes, counts_1, label='$\mu(s_1,a_2)$')
        plt.plot(replay_buffer_counts_episodes, counts_2, label='$\mu(s_2, a_1$)')
        #plt.plot(replay_buffer_counts_episodes, counts_3, label='mu_s2_a2')

        plt.ylabel('Probability')
        plt.xlabel('Episode')

        plt.ylim([0.0, 0.2])

        plt.legend()

        plt.savefig(f'{PLOTS_FOLDER_PATH}/{k}_replay_buffer_counts.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{PLOTS_FOLDER_PATH}/{k}_replay_buffer_counts.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

    """
        Correct actions.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (d, lbl) in zip(data, LABELS):
        weights_mean = np.mean(d['weights'], axis=0) # [E,F]

        c_1 = (weights_mean[:,0] > weights_mean[:,1]) # [E]
        c_2 = (1.2*weights_mean[:,0] < weights_mean[:,2]) # [E]

        correct_actions = c_1.astype(int) + c_2.astype(int) # [E]   

        plt.plot(correct_actions, label=lbl)

    plt.ylim([-0.1, 2.1])
    plt.yticks([0.0, 1.0, 2.0], ['0', '1', '2'])
    plt.ylabel('\# correct actions')
    plt.xlabel('Episode')
    plt.legend()

    plt.savefig(f'{PLOTS_FOLDER_PATH}/correct_actions_different_dists.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{PLOTS_FOLDER_PATH}/correct_actions_different_dists.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
