import os
import sys
import math
import json
import tarfile
import numpy as np
import pathlib
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


FIGURE_X = 6.0
FIGURE_Y = 4.0

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiments id.', required=True)
    parser.add_argument('--val_iter_exp', help='Value iteration experiments id.', required=True)

    return parser.parse_args()

# def calculate_CI_bootstrap(x_hat, samples, num_resamples=20000):
#     """
#         Calculates 95 % interval using bootstrap.
#         REF: https://ocw.mit.edu/courses/mathematics/
#             18-05-introduction-to-probability-and-statistics-spring-2014/
#             readings/MIT18_05S14_Reading24.pdf
#     """
#     resampled = np.random.choice(samples,
#                                 size=(len(samples), num_resamples),
#                                 replace=True)
#     means = np.mean(resampled, axis=0)
#     diffs = means - x_hat
#     bounds = [x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95)]
#
#     return bounds

def get_args_json_file(path):
    with open(path + "/args.json", 'r') as f:
        args = json.load(f)
    f.close()
    return args

def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError()

def print_env(data, sizes, float_format=None):
    size_x, size_y = sizes[0], sizes[1]
    sys.stdout.write('-'*(size_x+2)+'\n')
    for h in range(size_y):
        sys.stdout.write('|')
        for w in range(size_x):
            if float_format:
                sys.stdout.write(float_format.format(data[xy_to_idx((w,h),size_x, size_y)]))
            else:
                sys.stdout.write(str(data[xy_to_idx((w,h), size_x, size_y)]))
        sys.stdout.write('|\n')
    sys.stdout.write('-' * (size_x + 2)+'\n')

# def ridgeline(x, data, ax, overlap=0, fill=True, labels=None, n_points=150):
#     if overlap > 1 or overlap < 0:
#         raise ValueError('overlap must be in [0 1]')
#     # xx = np.linspace(np.min(np.concatenate(data)),
#     #                  np.max(np.concatenate(data)), n_points)
#     # curves = []
#     # ys = []
#     for i, d in enumerate(data):
#         # pdf = gaussian_kde(d)
#         y = i*(1.0-overlap)
#         # ys.append(y)
#         # curve = pdf(xx)
#         # if fill:
#         #     ax.fill_between(xx, np.ones(n_points)*y, 
#         #                      curve+y, zorder=len(data)-i+1, color=fill)
#         ax.plot(x, d+y, c='k', zorder=len(data)-i+1)
#     # if labels:
#     #     ax.yticks(ys, labels)


def main(exp_id, val_iter_exp):

    # Parse arguments if needed.
    if (exp_id is None) or (val_iter_exp is None):
        c_args = get_arguments()
        exp_id = c_args.exp
        val_iter_exp = c_args.val_iter_exp

    print('Arguments (analysis/plots.py):')
    print('Exp. id: {0}'.format(exp_id))
    print('Val. iter. exp. id: {0}\n'.format(val_iter_exp))

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/'
    os.makedirs(output_folder, exist_ok=True)
    q_vals_folder_path = PLOTS_FOLDER_PATH + exp_id + '/Q-values'
    os.makedirs(q_vals_folder_path, exist_ok=True)

    # Get args file (assumes all experiments share the same arguments).


    # Load optimal policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + val_iter_exp
    print(f"Opening experiment {val_iter_exp}")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        val_iter_data = json.load(f)
        val_iter_data = json.loads(val_iter_data)
        val_iter_data = val_iter_data
    f.close()
    val_iter_data['Q_vals'] = np.array(val_iter_data['Q_vals']) # [S,A]

    # Open data.
    print(f"Opening experiment {exp_id}")
    if exp_id[-7:] == '.tar.gz':
        exp_folder_path = DATA_FOLDER_PATH + exp_id

        tar = tarfile.open(exp_folder_path)
        data_file = tar.extractfile("{0}/train_data.json".format(exp_id))

        exp_data = json.load(data_file)
        exp_data = json.loads(exp_data)

        args_file = tar.extractfile("{0}/args.json".format(exp_id))
        exp_args = json.load(args_file)
        exp_args = json.loads(exp_args)

    else:
        exp_path = DATA_FOLDER_PATH + exp_id
        with open(exp_path + "/train_data.json", 'r') as f:
            exp_data = json.load(f)
            exp_data = json.loads(exp_data)
        f.close()

        exp_args = get_args_json_file(DATA_FOLDER_PATH + exp_id)
    
    # Store a copy of the args.json file inside plots folder.
    with open(output_folder + "args.json", 'w') as f:
        json.dump(exp_args, f)
        f.close()
    print('Exp. args:')
    print(exp_args)

    # Parse data.
    data = {}

    # steps field.
    data['steps'] = exp_data[0]['steps'] # [(Steps)]

    # Q_vals field.
    data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,(Steps),S,A]

    # rollouts_rewards field.
    data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(Steps)]

    # Scalar metrics dict.
    scalar_metrics = {}

    """
        `rollouts_rewards` plot.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (i, run_rollouts) in enumerate(data['rollouts_rewards']): # run_rollouts = [(Steps)]
        X = data['steps'] # [(Steps)]
        Y = run_rollouts

        plt.plot(X, Y, label=f'Run {i}')

    plt.xlabel('Learning step')
    plt.ylabel('Rollouts average reward')
    plt.legend()

    plt.savefig('{0}/rollouts_rewards.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Scalar evaluation rewards metrics.
    """
    scalar_metrics[f'rollouts_rewards_final'] = np.mean(data['rollouts_rewards'][:,-1])

    """
        `maximizing_action_s_*` plots.
    """
    """ if exp_args['env_name'] in ('gridEnv1', 'gridEnv2'):
        print('Computing `maximizing_action_s_*` plots.')

        for state in range(data['Q_vals'].shape[2]):

            num_cols = 3
            num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_data) in zip(axs.flat, data['Q_vals']): # run_data = [(Steps),S,A]

                state_data = run_data[:,state,:] # [(Steps),A]
                Y = np.argmax(state_data, axis=1) # [(Steps)]
                X = data['steps']

                ax.plot(X, Y)

                ax.set_ylim([-0.1,data['Q_vals'].shape[-1]+0.1])
                ax.set_ylabel('Maximizing action')
                ax.set_xlabel('Learning step')
                ax.set_title(f'Run {i}')

                i += 1

            fig.suptitle(f'Maximizing actions: state {state}')

            plt.savefig(f'{q_vals_folder_path}/maximizing_action_s_{state}.png', bbox_inches='tight', pad_inches=0)
            plt.close() """

    """
        `q_values_summed_error` plot.
    """
    print('Computing `q_values_summed_error` plot.')
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    
    for (i, run_Q_vals) in enumerate(data['Q_vals']): # run_Q_vals = [(Steps),S,A]
        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [(Steps),S,A]
        Y = np.sum(errors, axis=(1,2)) # [(Steps)]
        X = data['steps']

        plt.plot(X, Y, label=f'Run {i}')

    plt.xlabel('Learning step')
    plt.ylabel('Q-values error')
    plt.title('sum(abs(val_iter_Q_vals - Q_vals))')
    plt.legend()

    plt.savefig('{0}/q_values_summed_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        Scalar q-values metrics.
    """
    errors = np.abs(val_iter_data['Q_vals'] - data['Q_vals']) # [R,(Steps),S,A]
    errors = np.sum(errors, axis=(2,3)) # [R,(Steps)]
    scalar_metrics['qvals_summed_error'] = np.mean(errors[:,-1])

    errors = np.abs(val_iter_data['Q_vals'] - data['Q_vals']) # [R,(Steps),S,A]
    errors = np.mean(errors, axis=(2,3)) # [R,(Steps)]
    scalar_metrics['qvals_avg_error'] = np.mean(errors[:,-1])

    """
        `q_values_mean_error` plot.
    """
    print('Computing `q_values_mean_error` plot.')
    num_cols = 3
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [(Steps),S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [(Steps),S,A]
        Y = np.mean(errors, axis=(1,2)) # [(Steps)]
        X = data['steps']
        # Y_std = np.std(errors, axis=(1,2))

        # CI calculation.
        # errors_flatten = errors.reshape(len(Y), -1) # [(Steps),S*A]
        # X_CI = np.arange(0, len(Y), 100)
        # CI_bootstrap = [calculate_CI_bootstrap(Y[x], errors_flatten[x,:]) for x in X_CI]
        # CI_bootstrap = np.array(CI_bootstrap).T
        # CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        # CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

        p = ax.plot(X,Y,label='Q-vals')
        # ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)
        # ax.fill_between(X, Y-Y_std, Y+Y_std, color=p[0].get_color(), alpha=0.15)

        # Max Q-values.
        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [(Steps),S,A]
        maximizing_actions = np.argmax(run_Q_vals, axis=2) # [(Steps),S]
        x, y = np.indices(maximizing_actions.shape)
        errors = errors[x,y,maximizing_actions] # [(Steps),S]

        Y = np.mean(errors, axis=1) # [(Steps)]
        X = data['steps']
        # CI calculation.
        # X_CI = np.arange(0, len(Y), 100)
        # CI_bootstrap = [calculate_CI_bootstrap(Y[x], errors[x,:]) for x in X_CI]
        # CI_bootstrap = np.array(CI_bootstrap).T
        # CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        # CI_lengths = np.abs(np.subtract(CI_bootstrap,Y[X_CI]))

        p = ax.plot(X, Y, label='Max Q-vals')
        # ax.fill_between(X_CI, Y[X_CI]-CI_lengths[0], Y[X_CI]+CI_lengths[1], color=p[0].get_color(), alpha=0.15)

        #ax.set_ylim(y_axis_range)
        ax.set_ylabel('Q-values error')
        ax.set_xlabel('Learning step')
        ax.set_title(f'Run {i}')

        i += 1

    plt.legend()

    plt.savefig('{0}/q_values_mean_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        `q_values_violinplot_error` plot.
    """
    print('Computing `q_values_violinplot_error` plot.')
    num_cols = 3
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [(Steps),S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [(Steps),S,A]
        errors = errors.reshape(errors.shape[0], -1) # [(Steps),S*A]

        X = data['steps']
        violin = ax.violinplot(errors.tolist(), positions=X,
                                showextrema=True, widths=350)

        abs_diff_mean = np.mean(errors, axis=1)
        ax.scatter(X, abs_diff_mean, s=12)

        #ax.set_ylim(y_axis_range)
        ax.set_xlabel('Learning step')
        ax.set_ylabel('Q-values error')
        ax.set_title(label=f"Run {i}")

        i += 1

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals)')

    plt.savefig('{0}/q_values_violinplot_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        `q_values_violinplot_maxQ_error` plot.
    """
    print('Computing `q_values_violinplot_maxQ_error` plot.')
    num_cols = 3
    num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
    fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

    i = 0
    for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [(Steps),S,A]

        errors = np.abs(val_iter_data['Q_vals'] - run_Q_vals) # [(Steps),S,A]
        maximizing_actions = np.argmax(run_Q_vals, axis=2) # [(Steps),S]
        x, y = np.indices(maximizing_actions.shape)
        errors = errors[x,y,maximizing_actions] # [(Steps),S]

        X = data['steps']
        violin = ax.violinplot(errors.tolist(), positions=X,
                                showextrema=True, widths=350)

        abs_diff_mean = np.mean(errors, axis=1)
        ax.scatter(X, abs_diff_mean, s=12)

        #ax.set_ylim(y_axis_range)
        ax.set_xlabel('Learning step')
        ax.set_ylabel('Q-values error')
        ax.set_title(label=f"Run {i}")

        i += 1

    fig.suptitle('Dist. of abs(val_iter_Q_vals - Q_vals) for a=argmax(Q_vals)')
    plt.legend()

    plt.savefig('{0}/q_values_violinplot_maxQ_error.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    """
        `Q_values_s*-*-*` plots.
    """
    print('Computing `Q_values_s*-*-*` plots.')
    if exp_args['env_name'] in ('gridEnv1', 'gridEnv2'):
        for state_to_plot in range(data['Q_vals'].shape[2]):

            # Plot.
            num_cols = 3
            num_rows = math.ceil(data['Q_vals'].shape[0] / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.subplots_adjust(top=0.92, wspace=0.18, hspace=0.3)
            fig.set_size_inches(FIGURE_X*num_cols, FIGURE_Y*num_rows)

            i = 0
            for (ax, run_Q_vals) in zip(axs.flat, data['Q_vals']): # run_Q_vals = [(Steps),S,A]

                for action in range(run_Q_vals.shape[-1]):
                    
                    # Plot predicted Q-values.
                    Y = run_Q_vals[:,state_to_plot,action]
                    X = data['steps']
                    l = ax.plot(X, Y, label=f'Action {action}')

                    # Plot true Q-values.
                    ax.hlines(val_iter_data['Q_vals'][state_to_plot,action],
                            xmin=0, xmax=max(data['steps']), linestyles='--', color=l[0].get_color())

                #ax.set_ylim(y_axis_range)
                ax.set_xlabel('Learning step')
                ax.set_ylabel('Q-value')
                ax.set_title(label=f"Run {i}")
                ax.legend()

                i += 1

            fig.suptitle(f'Q-values: state {state_to_plot}')

            plt.savefig(f'{q_vals_folder_path}/Q_values_s{state_to_plot}.png', bbox_inches='tight', pad_inches=0)

            plt.close()

    # Store scalars dict.
    with open(output_folder + "scalar_metrics.json", 'w') as f:
        json.dump(scalar_metrics, f)
        f.close()

if __name__ == "__main__":
    main(exp_id=None, val_iter_exp=None)
