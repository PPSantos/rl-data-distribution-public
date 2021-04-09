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

EXP_IDS = ['8_8_q_learning_2021-04-09-18-04-25']

VAL_ITER_DATA = '8_8_val_iter_2021-04-09-18-08-36'


if __name__ == "__main__":

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Load data.
    data = {}
    for exp_name in EXP_IDS:

        exp_path = DATA_FOLDER_PATH + exp_name
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            exp_data = json.loads(exp_data)
        f.close()

        # Average data for each train run.
        averaged_data = {}

        # episode_rewards
        r_concatenated = np.array([exp['episode_rewards'] for exp in exp_data])
        r_concatenated = np.mean(r_concatenated, axis=0)
        averaged_data['episode_rewards'] = r_concatenated

        # epsilon_values
        e_concatenated = np.array([exp['epsilon_values'] for exp in exp_data])
        e_concatenated = np.mean(e_concatenated, axis=0)
        averaged_data['epsilon_values'] = e_concatenated

        # Q_vals
        q_concatenated = np.array([exp['Q_vals'] for exp in exp_data])
        q_concatenated = np.mean(q_concatenated, axis=0)
        averaged_data['Q_vals'] = q_concatenated

        data[exp_name] = averaged_data

    if VAL_ITER_DATA:
        # Load optimal Q-values.
        val_iter_path = DATA_FOLDER_PATH + VAL_ITER_DATA
        with open(val_iter_path + "/train_data.json", 'r') as f:
            val_iter_data = json.load(f)
            val_iter_data = json.loads(val_iter_data)
            val_iter_data = val_iter_data[0]
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

    """
        Q-values (summed-error).
    """
    if VAL_ITER_DATA:

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        val_iter_Q_vals = np.array(val_iter_data['Q_vals'])

        for exp_id in EXP_IDS:

            exp_Q_vals = np.array(data[exp_id]['Q_vals'])

            Y = np.sum(np.abs(val_iter_Q_vals - exp_Q_vals), axis=(1,2))
            X = np.linspace(1, len(Y), len(Y))

            plt.plot(X,Y,label=exp_id)

        plt.xlabel('Episode')
        plt.ylabel('Sum error')
        plt.title('np.sum(np.abs(val_iter_Q_vals - exp_Q_vals), axis=(1,2))')

        plt.savefig('{0}/q_values_summed_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/q_values_summed_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.close()

    """
        Q-values (mean-error).
    """
    if VAL_ITER_DATA:

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        val_iter_Q_vals = np.array(val_iter_data['Q_vals'])

        for exp_id in EXP_IDS:

            exp_Q_vals = np.array(data[exp_id]['Q_vals'])

            Y = np.mean(np.abs(val_iter_Q_vals - exp_Q_vals), axis=(1,2))
            X = np.linspace(1, len(Y), len(Y))

            plt.plot(X,Y,label=exp_id)

        plt.xlabel('Episode')
        plt.ylabel('Mean error')
        plt.title('np.mean(np.abs(val_iter_Q_vals - exp_Q_vals), axis=(1,2))')

        plt.savefig('{0}/q_values_mean_error.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/q_values_mean_error.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
        plt.close()

    """ def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError() """

    """ #print('\nQ-vals:', Q_vals)
        print('Max Q-vals:', max_Q_vals)
        print('Policy:', policy)
        env.render()

        # Plot policy.
        print('Policy:')
        size_x = args['env_args']['size_x']
        size_y = args['env_args']['size_y']
        sys.stdout.write('-'*(size_x+2)+'\n')
        for h in range(size_y):
            sys.stdout.write('|')
            for w in range(size_x):
                sys.stdout.write(str(policy[xy_to_idx((w,h), size_x, size_y)]))
            sys.stdout.write('|\n')
        sys.stdout.write('-' * (size_x + 2)+'\n')

        # Plot max Q-values.
        print('Max Q-vals:')
        for h in range(size_y):
            for w in range(size_x):
                sys.stdout.write("{:.1f} ".format(max_Q_vals[xy_to_idx((w,h),size_x, size_y)]))
            sys.stdout.write('\n')
        sys.stdout.write('\n')
    """
