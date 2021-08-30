import os
import sys
import math
import json
import random
import numpy as np
import pathlib
import tarfile

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import statsmodels.api as sm
from scipy import stats
import statsmodels
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

from envs import env_suite

FIGURE_X = 6.0
FIGURE_Y = 4.0

GRAY_COLOR = (0.3,0.3,0.3)

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/compare/'


# SETUP VARIABLES.
ENV_NAME = 'gridEnv1'
VAL_ITER_DATA = 'gridEnv1_val_iter_2021-05-14-15-54-10'
EXPS_DATA = [
            {'id': 'gridEnv1_dqn_e_func_2021-08-25-16-42-24.tar.gz', 'label': 'delta=0.0'},
            {'id': 'gridEnv1_dqn_e_func_2021-08-25-19-52-29.tar.gz', 'label': 'delta=0.2'},
            {'id': 'gridEnv1_dqn_e_func_2021-08-25-23-04-01.tar.gz', 'label': 'delta=0.4'},
            {'id': 'gridEnv1_dqn_e_func_2021-08-26-02-17-03.tar.gz', 'label': 'delta=0.6'},
            {'id': 'gridEnv1_dqn_e_func_2021-08-26-05-30-13.tar.gz', 'label': 'delta=0.8'},
            {'id': 'gridEnv1_dqn_e_func_2021-08-26-08-42-05.tar.gz', 'label': 'delta=1.0'},
            ]

EXPS_DATA_2 = None


def calculate_CI_bootstrap(x_hat, samples, num_resamples=20_000):
    """
        Calculates 95 % interval using bootstrap.
        REF: https://ocw.mit.edu/courses/mathematics/
            18-05-introduction-to-probability-and-statistics-spring-2014/
            readings/MIT18_05S14_Reading24.pdf
    """
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    means = np.mean(resampled, axis=0)
    diffs = means - x_hat
    bounds = [x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95)]
    return bounds


def main():

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

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
    for exp in EXPS_DATA:

        exp_path = DATA_FOLDER_PATH + exp['id']
        print(f"Opening experiment {exp['id']}")
        if pathlib.Path(exp_path).suffix == '.gz':

            exp_name = pathlib.Path(exp_path).stem
            exp_name = '.'.join(exp_name.split('.')[:-1])
            print('exp_name', exp_name)

            tar = tarfile.open(exp_path)
            data_file = tar.extractfile("{0}/train_data.json".format(exp_name))

            exp_data = json.load(data_file)
            exp_data = json.loads(exp_data)

        else:
            with open(exp_path + "/train_data.json", 'r') as f:
                exp_data = json.load(f)
                exp_data = json.loads(exp_data)
            f.close()

        # Parse data for each train run.
        parsed_data = {}

        # Q_vals field.
        parsed_data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]

        # rollouts_rewards field.
        parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]

        data[exp['id']] = parsed_data

    # Load and parse additional data.
    if EXPS_DATA_2:
        for exp, exp_2 in zip(EXPS_DATA, EXPS_DATA_2):

            exp_path = DATA_FOLDER_PATH + exp_2['id']
            print(f"Opening experiment {exp_2['id']}")
            if pathlib.Path(exp_path).suffix == '.gz':

                exp_name = pathlib.Path(exp_path).stem
                exp_name = '.'.join(exp_name.split('.')[:-1])
                print('exp_name', exp_name)

                tar = tarfile.open(exp_path)
                data_file = tar.extractfile("{0}/train_data.json".format(exp_name))

                exp_data = json.load(data_file)
                exp_data = json.loads(exp_data)

            else:
                with open(exp_path + "/train_data.json", 'r') as f:
                    exp_data = json.load(f)
                    exp_data = json.loads(exp_data)
                f.close()

            # Parse data for each train run.
            parsed_data = {}

            # Q_vals field.
            parsed_data['Q_vals'] = np.array([e['Q_vals'] for e in exp_data]) # [R,E,S,A]
            data[exp['id']]['Q_vals'] = np.concatenate([data[exp['id']]['Q_vals'], parsed_data['Q_vals']])

            # rollouts_rewards field.
            parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards'] for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]
            data[exp['id']]['rollouts_rewards'] = np.concatenate([data[exp['id']]['rollouts_rewards'], parsed_data['rollouts_rewards']])

    # Load additional variables from last experiment file.
    Q_vals_episodes = exp_data[0]['Q_vals_episodes'] # [(E)]
    rollouts_episodes = exp_data[0]['rollouts_episodes'] # [(E)]

    """
        Q-values errors.
    """
    print('-'*20)
    print('Q-values:')
    print('-'*20)

    # Mean Q-values error throughout training
    # (with bootstrapped confidence interval).
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for exp in EXPS_DATA:
        exp_data = data[exp['id']]

        # Calculate and plot mean value throughout episodes.
        errors = np.abs(val_iter_data['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
        errors = np.mean(errors, axis=0) # [E,S,A]
        Y = np.mean(errors, axis=(1,2)) # [E]
        X = Q_vals_episodes
        p = plt.plot(X, Y, label=exp['label'])

        # Calculate and plot mean confidence interval.
        ci_errors = np.abs(val_iter_data['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
        ci_errors = np.mean(ci_errors, axis=(2,3)) # [R,E]
        CI_bootstrap = [calculate_CI_bootstrap(Y[e], ci_errors[:,e])
                            for e in range(len(Y))]
        CI_bootstrap = np.array(CI_bootstrap).T
        CI_bootstrap = np.flip(CI_bootstrap, axis=0)
        CI_lengths = np.abs(np.subtract(CI_bootstrap,Y))
        plt.fill_between(X, Y-CI_lengths[0], Y+CI_lengths[1], color=p[0].get_color(), alpha=0.15)

    plt.xlabel('Episode')
    plt.ylabel('Q-values error')
    plt.title('avg(abs(val_iter_Q_vals - Q_vals))')

    plt.legend()

    #plt.savefig('{0}/qvals_avg_error_episodes.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/qvals_avg_error_episodes.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()


    # Means distribution plot for the last episode(s).
    errors_list = []
    for exp in EXPS_DATA:
        exp_data = data[exp['id']]
        errors = np.abs(val_iter_data['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
        errors = errors[:,-10:,:,:] # [R,(E),S,A]
        errors = np.mean(errors, axis=(1,2,3)) # [R]
        errors_list.append(errors)

        print(f'{exp["label"]} (last episode): {np.mean(errors)} ')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    x_ticks_pos = np.arange(1,len(EXPS_DATA)+1)
    plt.violinplot(errors_list, positions=x_ticks_pos, showextrema=True)

    for i in range(len(EXPS_DATA)):
        plt.scatter([x_ticks_pos[i]]*len(errors_list[i]), errors_list[i], color=GRAY_COLOR, zorder=100, alpha=0.6)

    plt.xlabel('Algorithm')
    plt.ylabel('Q-values error')
    plt.title('avg(abs(val_iter_Q_vals - Q_vals))')

    plt.xticks(ticks=x_ticks_pos, labels=[e['label'] for e in EXPS_DATA])

    #plt.savefig('{0}/qvals_avg_error_final.pdf'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/qvals_avg_error_final.png'.format(PLOTS_FOLDER_PATH), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Statistical tests for last episode(s) Q-values errors.
    processed_data = {}
    for exp in EXPS_DATA:
        exp_data = data[exp['id']]
        errors = np.abs(val_iter_data['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
        errors = errors[:,-10:,:,:] # [R,(E),S,A]
        errors = np.mean(errors, axis=(1,2,3)) # [R]
        processed_data[exp['id']] = errors

    # Shapiro test (asserts that data is normally distributed for all groups).
    print('Shapiro tests:')
    for exp in EXPS_DATA:
        shapiro_test = stats.shapiro(processed_data[exp['id']])
        print(f'\t{exp["label"]}: {shapiro_test}')

    # Parametric Levene test (asserts that all groups share the same variance).
    t_args = [processed_data[exp['id']] for exp in EXPS_DATA]
    print(f'\nLevene\'s test: {stats.levene(*t_args)}')

    # Parametric ANOVA test (test whether all groups share the same mean value or not).
    print(f'\nANOVA test: {stats.f_oneway(*t_args)}')

    # Tukey HSD test (pairwise comparisons between all groups).
    t_data, groups = [], []
    for exp in EXPS_DATA:
        groups.extend([exp['label'] for _ in range(len(processed_data[exp['id']]))])
        t_data.extend(processed_data[exp['id']])
    print('\nTukeyHSD:', pairwise_tukeyhsd(t_data, groups))

    # Non-parametric test.
    print('\nKruskal (non-parametric) test:', stats.kruskal(*t_args))

    # Post-hoc non-parametric comparisons.
    t_data = [processed_data[exp['id']] for exp in EXPS_DATA]
    print(sp.posthoc_conover(t_data, p_adjust='holm'))


    """
        Rollouts rewards.
    """
    if ENV_NAME in env_suite.CUSTOM_GRID_ENVS.keys():
        rollouts_types = sorted(env_suite.CUSTOM_GRID_ENVS[ENV_NAME].keys())
    elif ENV_NAME == 'pendulum':
        rollouts_types = sorted(env_suite.PENDULUM_ENVS.keys())
    elif ENV_NAME == 'mountaincar':
        rollouts_types = sorted(env_suite.MOUNTAINCAR_ENVS.keys())
    elif ENV_NAME == 'multiPathsEnv':
        rollouts_types = sorted(env_suite.MULTIPATHS_ENVS.keys())
    else:
        raise ValueError(f'Env. {ENV_NAME} does not have rollout types defined.')

    for t, rollout_type in enumerate(rollouts_types):

        print('-'*20)
        print(f'Rollouts ({rollout_type}):')
        print('-'*20)

        # Average plot throughout training
        # (with bootstrapped confidence interval).
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for exp in EXPS_DATA:
            exp_data = data[exp['id']]
            rollout_type_data = exp_data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]
            rollout_averaged_per_run = np.average(rollout_type_data, axis=2) # [R,(E)]
            rollout_averaged = np.average(rollout_averaged_per_run, axis=0) # [(E)]
            Y = rollout_averaged
            X = rollouts_episodes
            p = plt.plot(X, Y, label=exp['label'])

            # Calculate and plot confidence interval.
            ci_errors = rollout_averaged_per_run # [R,(E)]
            CI_bootstrap = [calculate_CI_bootstrap(Y[e], ci_errors[:,e])
                                for e in range(len(Y))]
            CI_bootstrap = np.array(CI_bootstrap).T
            CI_bootstrap = np.flip(CI_bootstrap, axis=0)
            CI_lengths = np.abs(np.subtract(CI_bootstrap,Y))
            plt.fill_between(X, Y-CI_lengths[0], Y+CI_lengths[1], color=p[0].get_color(), alpha=0.15)

        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.legend()

        #plt.savefig('{0}/{1}_episodes.pdf'.format(PLOTS_FOLDER_PATH, rollout_type), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/{1}_episodes.png'.format(PLOTS_FOLDER_PATH, rollout_type), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Means distribution plot (last episode(s)).
        errors_list = []
        for exp in EXPS_DATA:
            exp_data = data[exp['id']]
            rollout_type_data = exp_data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]
            rollout_averaged_per_run = np.average(rollout_type_data, axis=2) # [R,(E)]
            last_eps_data = rollout_averaged_per_run[:,-10:] # [R,(E)]
            last_eps_data = np.mean(last_eps_data, axis=1) # [R]
            errors_list.append(last_eps_data)

            print(f'{exp["label"]} (last episode(s)): {np.mean(last_eps_data)} ')

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        x_ticks_pos = np.arange(1,len(EXPS_DATA)+1)
        plt.violinplot(errors_list, positions=x_ticks_pos, showextrema=True)

        for i in range(len(EXPS_DATA)):
            plt.scatter([x_ticks_pos[i]]*len(errors_list[i]), errors_list[i], color=GRAY_COLOR, zorder=100, alpha=0.6)

        plt.xlabel('Algorithm')
        plt.ylabel('Reward')

        plt.xticks(ticks=x_ticks_pos, labels=[e['label'] for e in EXPS_DATA])

        #plt.savefig('{0}/{1}_final.pdf'.format(PLOTS_FOLDER_PATH,rollout_type), bbox_inches='tight', pad_inches=0)
        plt.savefig('{0}/{1}_final.png'.format(PLOTS_FOLDER_PATH,rollout_type), bbox_inches='tight', pad_inches=0)
        plt.close()

        # Statistical tests for last episode.
        processed_data = {}
        for exp in EXPS_DATA:
            exp_data = data[exp['id']]
            rollout_type_data = exp_data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]
            rollout_averaged_per_run = np.average(rollout_type_data, axis=2) # [R,(E)]
            processed_data[exp['id']] = np.mean(rollout_averaged_per_run[:,-10:], axis=1) # [R]

        # Shapiro test (asserts that data is normally distributed for all groups).
        print('Shapiro tests:')
        for exp in EXPS_DATA:
            shapiro_test = stats.shapiro(processed_data[exp['id']])
            print(f'\t{exp["label"]}: {shapiro_test}')

        # Parametric Levene test (asserts that all groups share the same variance).
        t_args = [processed_data[exp['id']] for exp in EXPS_DATA]
        print(f'\nLevene\'s test: {stats.levene(*t_args)}')

        # Parametric ANOVA test (test whether all groups share the same mean value or not).
        print(f'\nANOVA test: {stats.f_oneway(*t_args)}')

        # Tukey HSD test (pairwise comparisons between all groups).
        t_data, groups = [], []
        for exp in EXPS_DATA:
            groups.extend([exp['label'] for _ in range(len(processed_data[exp['id']]))])
            t_data.extend(processed_data[exp['id']])
        print('\nTukeyHSD:', pairwise_tukeyhsd(t_data, groups))

        # Non-parametric test.
        print('\nKruskal (non-parametric) test:', stats.kruskal(*t_args))

        # Post-hoc non-parametric comparisons.
        t_data = [processed_data[exp['id']] for exp in EXPS_DATA]
        print(sp.posthoc_conover(t_data, p_adjust='holm'))


if __name__ == "__main__":
    main()