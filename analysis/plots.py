import os
import json
import numpy as np
import pathlib

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'

EXP_IDS = ['8_8_q_learning_2021-04-09-12-27-16',
           '8_8_q_learning_2021-04-09-12-28-22']


if __name__ == "__main__":

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Load data.
    data = {}
    for exp_name in EXP_IDS:

        exp_path = DATA_FOLDER_PATH + exp_name
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            data[exp_name] = json.loads(exp_data)
        f.close()

    """
        Episode rewards.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp_id in EXP_IDS:
        Y = data[exp_id]['episode_rewards']
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X,Y,label=exp_id)

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.savefig('{0}/episode_rewards.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_rewards.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Epsilon.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp_id in EXP_IDS:
        Y = data[exp_id]['epsilon_values']
        X = np.linspace(1, len(Y), len(Y))

        plt.plot(X,Y,label=exp_id)

    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.savefig('{0}/episode_epsilon.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/episode_epsilon.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    print(data[EXP_IDS[0]]['policy'])
    print(data[EXP_IDS[1]]['policy'])
