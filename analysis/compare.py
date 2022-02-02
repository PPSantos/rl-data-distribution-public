import os
import json
import numpy as np
import pathlib
import tarfile
import scipy

import matplotlib

from scripts.run import ORACLE_Q_VALS_DATA
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

from analysis.rliable import library as rly

from envs import env_suite

#####################################################################
########################## SCRIPT ARGUMENTS #########################
#####################################################################
ENV_NAME = 'pendulum'
ORACLE_Q_VALS_DATA = 'pendulum_val_iter_2021-05-24-11-48-50'
EXPS_DATA = [
    {'id': 'pendulum_offline_dqn_2021-09-30-08-18-18.tar.gz', 'label_1': '6.6', 'label_2': r'$\mathbb{E}[\mathcal{H}(\mu)] = 6.6$'},
    #{'id': 'pendulum_offline_dqn_2021-10-09-00-50-27.tar.gz', 'label_1': '7.3', 'label_2': r'$\mathbb{E}[\mathcal{H}(\mu)] = 7.3$'},
    {'id': 'pendulum_offline_dqn_2021-09-30-10-51-08.tar.gz', 'label_1': '7.8', 'label_2': r'$\mathbb{E}[\mathcal{H}(\mu)] = 7.8$'},
    {'id': 'pendulum_offline_dqn_2021-09-30-12-07-27.tar.gz', 'label_1': '8.1', 'label_2': r'$\mathbb{E}[\mathcal{H}(\mu)] = 8.1$'},
    {'id': 'pendulum_offline_dqn_2021-09-30-13-23-51.tar.gz', 'label_1': '8.3', 'label_2': r'$\mathbb{E}[\mathcal{H}(\mu)] = 8.3$'},
    {'id': 'pendulum_offline_dqn_2021-09-30-16-07-30.tar.gz', 'label_1': '8.4', 'label_2': r'$\mathbb{E}[\mathcal{H}(\mu)] = 8.4$'},
]
EXPS_DATA_2 = None # Allows to concatenate additional experiments.
STAT_TESTS = False # Whether to compute statistical tests.

#####################################################################

FIGURE_X = 6.0
FIGURE_Y = 4.0
GRAY_COLOR = (0.3,0.3,0.3)

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/compare/'


def mean(samples: np.ndarray, num_resamples: int=25_000):
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

def median(samples: np.ndarray, num_resamples=25_000):
    """
        Computes median.
    """
    # Point estimation.
    point_estimate = np.median(samples)
    # Confidence interval estimation.
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = np.median(resampled, axis=0)
    confidence_interval = [np.percentile(point_estimations, 5),
                           np.percentile(point_estimations, 95)]
    return point_estimate, confidence_interval

def iqm(samples: np.ndarray, num_resamples=25_000):
    """
        Computes the interquartile mean.
    """
    # Point estimation.
    point_estimate = scipy.stats.trim_mean(samples, proportiontocut=0.25, axis=None)
    # Confidence interval estimation.
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = scipy.stats.trim_mean(resampled, proportiontocut=0.25, axis=0)
    confidence_interval = [np.percentile(point_estimations, 5),
                           np.percentile(point_estimations, 95)]
    return point_estimate, confidence_interval

def optimality_gap(samples: np.ndarray, threshold: float, num_resamples=25_000):
    """
        Computes the optimality gap.
    """
    # Point estimation.
    point_estimate = np.mean((samples < threshold))
    # Confidence interval estimation.
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = np.mean((resampled < threshold), axis=0)
    confidence_interval = [np.percentile(point_estimations, 5),
                           np.percentile(point_estimations, 95)]
    return point_estimate, confidence_interval


def main():

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Load optimal policy/Q-values.
    val_iter_path = DATA_FOLDER_PATH + ORACLE_Q_VALS_DATA
    print(f"Opening experiment {ORACLE_Q_VALS_DATA}")
    with open(val_iter_path + "/train_data.json", 'r') as f:
        oracle_q_vals_data = json.load(f)
        oracle_q_vals_data = json.loads(oracle_q_vals_data)
        oracle_q_vals_data = oracle_q_vals_data[0]
    f.close()
    oracle_q_vals_data['Q_vals'] = np.array(oracle_q_vals_data['Q_vals']) # [S,A]

    # Load and parse data.
    data = {}
    for exp in EXPS_DATA:

        exp_path = DATA_FOLDER_PATH + exp['id']
        print(f"Opening experiment {exp['id']}")
        if pathlib.Path(exp_path).suffix == '.gz':

            exp_name = pathlib.Path(exp_path).stem
            exp_name = '.'.join(exp_name.split('.')[:-1])

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
        parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards']
                        for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]

        data[exp['id']] = parsed_data

    # Load and parse additional data (if needed).
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
            data[exp['id']]['Q_vals'] = np.concatenate([data[exp['id']]['Q_vals'],
                                                        parsed_data['Q_vals']])

            # rollouts_rewards field.
            parsed_data['rollouts_rewards'] = np.array([e['rollouts_rewards']
                            for e in exp_data]) # [R,(E),num_rollouts_types,num_rollouts]
            data[exp['id']]['rollouts_rewards'] = np.concatenate([data[exp['id']]['rollouts_rewards'],
                                                        parsed_data['rollouts_rewards']])

    # Load additional variables from last experiment file.
    Q_vals_episodes = exp_data[0]['Q_vals_steps'] # [(E)]
    rollouts_episodes = exp_data[0]['rollouts_steps'] # [(E)]

    aggregate_funcs = {'mean': mean, 'median': median, 'iqm': iqm}

    """
        Q-values errors.
    """
    print('Computing Q-values errors plots...')

    # Pre-process Q-values data.
    q_vals_processed_data = {}
    for exp in EXPS_DATA:
        exp_data = data[exp['id']]
        errors = np.abs(oracle_q_vals_data['Q_vals'] - exp_data['Q_vals']) # [R,E,S,A]
        errors = np.mean(errors, axis=(2,3)) # [R,E]
        q_vals_processed_data[exp['id']] = errors

    # Q-values error throughout training (with bootstrapped confidence interval).
    for (func_lbl, agg_func) in aggregate_funcs.items():

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for exp in EXPS_DATA:
            errors = q_vals_processed_data[exp['id']] # [R,E]

            # Calculate for each episode.
            point_estimations, conf_intervals = [], []
            for episode in range(errors.shape[1]):
                point_est, c_int = agg_func(errors[:,episode])
                point_estimations.append(point_est)
                conf_intervals.append(c_int)
            conf_intervals = np.array(conf_intervals)

            # Plot.
            p = plt.plot(Q_vals_episodes, point_estimations, label=exp['label_2'])
            plt.fill_between(Q_vals_episodes, conf_intervals[:,0], conf_intervals[:,1],
                            color=p[0].get_color(), alpha=0.15)

        plt.xlabel('Learning step')
        plt.ylabel(r'$Q$-values error')
        plt.yscale('log')
        plt.legend()

        plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_episodes_{func_lbl}.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_episodes_{func_lbl}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    # Q-values error at the end of training (with bootstrapped confidence interval).
    for (func_lbl, agg_func) in aggregate_funcs.items():

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        ci_lower_bounds, ci_upper_bounds = [], []
        for exp_idx, exp in enumerate(EXPS_DATA):
            errors = q_vals_processed_data[exp['id']] # [R,E]
            errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)

            point_est, (lower_ci, upper_ci) = agg_func(errors)

            # Plot confidence interval.
            plt.bar(
                x=exp_idx,
                width=0.5,
                height=upper_ci - lower_ci,
                bottom=lower_ci,
                alpha=0.75,
                label=exp['label_1'])

            # Plot point estimate.
            plt.hlines(
                y=point_est,
                xmin=exp_idx - 0.25,
                xmax=exp_idx + 0.25,
                label=exp['label_1'],
                color='k',
                alpha=0.65)

            ci_lower_bounds.append(lower_ci)
            ci_upper_bounds.append(upper_ci)

        y_lim_lower = np.min(ci_lower_bounds) - \
                    (0.03*(np.max(ci_upper_bounds-np.min(ci_lower_bounds))))
        y_lim_lower = max(y_lim_lower, 0.001)
        y_lim_upper = np.max(ci_upper_bounds) + \
                    (0.03*(np.max(ci_upper_bounds-np.min(ci_lower_bounds))))
        plt.ylim(y_lim_lower, y_lim_upper)
        print('y_lim_lower', y_lim_lower)
        print('y_lim_upper', y_lim_upper)

        plt.xticks(list(range(len(EXPS_DATA))), [exp['label_1'] for exp in EXPS_DATA])
        plt.ylabel(r'$Q$-values error')
        plt.yscale('log')
        plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')

        plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_{func_lbl}.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_{func_lbl}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    # Q-values error distribution plot for the last episode(s).
    errors_list = []
    for exp in EXPS_DATA:
        errors = q_vals_processed_data[exp['id']] # [R,E]
        errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)
        errors_list.append(errors)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    x_ticks_pos = np.arange(1,len(EXPS_DATA)+1)
    plt.violinplot(errors_list, positions=x_ticks_pos, showextrema=True)

    for i in range(len(EXPS_DATA)):
        plt.scatter([x_ticks_pos[i]]*len(errors_list[i]), errors_list[i],
                        color=GRAY_COLOR, zorder=100, alpha=0.6)

    plt.ylabel(r'$Q$-values error')
    plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')
    plt.yscale('log')
    plt.xticks(ticks=x_ticks_pos, labels=[e['label_1'] for e in EXPS_DATA])

    plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_distribution.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_distribution.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # Performance profile plot for the last episode(s) (lower is better).
    scores_dict = {}
    max_q_vals_errors = []
    for exp in EXPS_DATA:
        errors = q_vals_processed_data[exp['id']] # [R,E]
        errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)
        max_q_vals_errors.append(np.max(errors))
        errors = errors[:,np.newaxis]
        scores_dict[exp['label_1']] = errors

    max_threshold = max(max_q_vals_errors) # Maximum Q-value error.
    thresholds = np.linspace(0.0, max_threshold, 101)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
                                                        scores_dict, thresholds)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # Plot.
    for exp in EXPS_DATA:
        score = score_distributions[exp['label_1']]
        lower_ci, upper_ci = score_distributions_cis[exp['label_1']]
        p = plt.plot(thresholds, score, label=exp['label_1'])
        plt.fill_between(thresholds, lower_ci, upper_ci,
                            color=p[0].get_color(), alpha=0.15)

    plt.xlabel(r'$Q$-values error $(\tau)$')
    plt.ylabel(r'Fraction of runs with error $> \tau$')
    plt.yscale('log')
    plt.legend()

    plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_performance_profile.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_performance_profile.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    if STAT_TESTS:
        # Statistical tests for last episode(s) Q-values errors.
        stat_test_data = {}
        for exp in EXPS_DATA:
            errors = q_vals_processed_data[exp['id']] # [R,E]
            errors = np.mean(errors[:,-10:], axis=1) # [R] (use last 10 episodes data)
            stat_test_data[exp['id']] = errors

        # Shapiro test (asserts that data is normally distributed for all groups).
        print('Shapiro tests:')
        for exp in EXPS_DATA:
            shapiro_test = stats.shapiro(stat_test_data[exp['id']])
            print(f'\t{exp["label"]}: {shapiro_test}')

        # Parametric Levene test (asserts that all groups share the same variance).
        t_args = [stat_test_data[exp['id']] for exp in EXPS_DATA]
        print(f'\nLevene\'s test: {stats.levene(*t_args)}')

        # Parametric ANOVA test (test whether all groups share the same mean value or not).
        print(f'\nANOVA test: {stats.f_oneway(*t_args)}')

        # Tukey HSD test (pairwise comparisons between all groups).
        t_data, groups = [], []
        for exp in EXPS_DATA:
            groups.extend([exp['label'] for _ in range(len(stat_test_data[exp['id']]))])
            t_data.extend(stat_test_data[exp['id']])
        print('\nTukeyHSD:', pairwise_tukeyhsd(t_data, groups))

        # Non-parametric test.
        print('\nKruskal (non-parametric) test:', stats.kruskal(*t_args))

        # Post-hoc non-parametric comparisons.
        t_data = [stat_test_data[exp['id']] for exp in EXPS_DATA]
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

    rollouts_types = ['default',]

    for t, rollout_type in enumerate(rollouts_types):
        print(f'Computing rollout {rollout_type} plots...')

        # Pre-process rollout rewards data.
        rollout_processed_data = {}
        for exp in EXPS_DATA:
            exp_data = data[exp['id']]
            rollout_data = exp_data['rollouts_rewards'][:,:,t,:] # [R,(E),num_rollouts]
            rollout_data = np.average(rollout_data, axis=2) # [R,(E)]
            rollout_processed_data[exp['id']] = rollout_data

        # Rollout reward during training (with bootstrapped confidence interval).
        for func_lbl, agg_func in aggregate_funcs.items():

            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            for exp in EXPS_DATA:
                rollout_data = rollout_processed_data[exp['id']] # [R,(E)]

                # Calculate for each episode.
                point_estimations, conf_intervals = [], []
                for episode in range(rollout_data.shape[1]):
                    point_est, c_int = agg_func(rollout_data[:,episode])
                    point_estimations.append(point_est)
                    conf_intervals.append(c_int)
                conf_intervals = np.array(conf_intervals)

                # Plot.
                p = plt.plot(rollouts_episodes, point_estimations, label=exp['label_2'])
                plt.fill_between(rollouts_episodes, conf_intervals[:,0], conf_intervals[:,1],
                                color=p[0].get_color(), alpha=0.15)

            plt.xlabel('Learning step')
            plt.ylabel('Reward')
            plt.legend()

            plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_episodes_{func_lbl}.pdf',
                        bbox_inches='tight', pad_inches=0)
            plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_episodes_{func_lbl}.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        # Rollout reward at the end of training (with bootstrapped confidence interval).
        for (func_lbl, agg_func) in aggregate_funcs.items():

            fig = plt.figure()
            fig.set_size_inches(FIGURE_X, FIGURE_Y)

            ci_lower_bounds, ci_upper_bounds = [], []
            for exp_idx, exp in enumerate(EXPS_DATA):
                rollout_data = rollout_processed_data[exp['id']] # [R,(E)]
                rollout_data = rollout_data[:,-10:] # [R,(E)]
                rollout_data = np.mean(rollout_data, axis=1) # [R]

                point_est, (lower_ci, upper_ci) = agg_func(rollout_data)

                # Plot confidence interval.
                plt.bar(
                    x=exp_idx,
                    width=0.5,
                    height=upper_ci - lower_ci,
                    bottom=lower_ci,
                    alpha=0.75,
                    label=exp['label_1'])

                # Plot point estimate.
                plt.hlines(
                    y=point_est,
                    xmin=exp_idx - 0.25,
                    xmax=exp_idx + 0.25,
                    label=exp['label_1'],
                    color='k',
                    alpha=0.65)

                ci_lower_bounds.append(lower_ci)
                ci_upper_bounds.append(upper_ci)

            y_lim_lower = np.min(ci_lower_bounds) - \
                        (0.03*(np.max(ci_upper_bounds-np.min(ci_lower_bounds))))
            y_lim_lower = max(y_lim_lower, 0.0)
            y_lim_upper = np.max(ci_upper_bounds) + \
                        (0.03*(np.max(ci_upper_bounds-np.min(ci_lower_bounds))))
            plt.ylim(y_lim_lower, y_lim_upper)

            plt.xticks(list(range(len(EXPS_DATA))), [exp['label_1'] for exp in EXPS_DATA])
            plt.ylabel('Reward')
            plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')

            plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_{func_lbl}.pdf',
                        bbox_inches='tight', pad_inches=0)
            plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_{func_lbl}.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        # Calculate max reward (for optimality gap calculation).
        max_rewards = []
        for exp in EXPS_DATA:
            rollout_data = rollout_processed_data[exp['id']] # [R,(E)]
            rollout_data = rollout_data[:,-10:] # [R,(E)]
            rollout_data = np.mean(rollout_data, axis=1) # [R]
            max_rewards.append(np.max(rollout_data))
        max_reward = max(max_rewards)

        # Optimality gap (the number of runs that fail to reach tau*max_reward).
        tau = 0.5

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        # Calculate optimality gap.
        ci_lower_bounds, ci_upper_bounds = [], []
        for exp_idx, exp in enumerate(EXPS_DATA):
            rollout_data = rollout_processed_data[exp['id']] # [R,(E)]
            rollout_data = rollout_data[:,-10:] # [R,(E)]
            rollout_data = np.mean(rollout_data, axis=1) # [R]

            point_est, (lower_ci, upper_ci) = optimality_gap(rollout_data,
                                                threshold=tau*max_reward)

            # Plot confidence interval.
            plt.bar(
                x=exp_idx,
                width=0.5,
                height=upper_ci - lower_ci,
                bottom=lower_ci,
                alpha=0.75,
                label=exp['label_1'])

            # Plot point estimate.
            plt.hlines(
                y=point_est,
                xmin=exp_idx - 0.25,
                xmax=exp_idx + 0.25,
                label=exp['label_1'],
                color='k',
                alpha=0.65)

            ci_lower_bounds.append(lower_ci)
            ci_upper_bounds.append(upper_ci)

        y_lim_lower = np.min(ci_lower_bounds) - \
                    (0.03*(np.max(ci_upper_bounds-np.min(ci_lower_bounds))))
        y_lim_lower = max(y_lim_lower, 0.0)
        y_lim_upper = np.max(ci_upper_bounds) + \
                    (0.03*(np.max(ci_upper_bounds-np.min(ci_lower_bounds))))
        plt.ylim(y_lim_lower, y_lim_upper)

        plt.xticks(list(range(len(EXPS_DATA))), [exp['label_1'] for exp in EXPS_DATA])
        plt.ylabel('Optimality gap')
        plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')

        plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_optimality_gap.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_optimality_gap.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # Rollout rewards distribution plot (last episode(s)).
        errors_list = []
        for exp in EXPS_DATA:
            rollout_data = rollout_processed_data[exp['id']] # [R,(E)]
            last_eps_data = rollout_data[:,-10:] # [R,(E)]
            last_eps_data = np.mean(last_eps_data, axis=1) # [R]
            errors_list.append(last_eps_data)

        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        x_ticks_pos = np.arange(1,len(EXPS_DATA)+1)
        plt.violinplot(errors_list, positions=x_ticks_pos, showextrema=True)

        for i in range(len(EXPS_DATA)):
            plt.scatter([x_ticks_pos[i]]*len(errors_list[i]), errors_list[i],
                        color=GRAY_COLOR, zorder=100, alpha=0.6)

        plt.ylabel('Reward')
        plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')

        plt.xticks(ticks=x_ticks_pos, labels=[e['label_1'] for e in EXPS_DATA])

        plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_distribution.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_distribution.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # Rollout rewards performance profile plot for the last episode(s).
        scores_dict = {}
        max_rewards = []
        for exp in EXPS_DATA:
            rollout_data = rollout_processed_data[exp['id']] # [R,(E)]
            last_eps_data = rollout_data[:,-10:] # [R,(E)]
            last_eps_data = np.mean(last_eps_data, axis=1) # [R]
            last_eps_data = last_eps_data[:,np.newaxis]
            max_rewards.append(np.max(last_eps_data))
            scores_dict[exp['label_1']] = last_eps_data

        max_threshold = max(max_rewards)
        thresholds = np.linspace(0.0, max_threshold, 501)
        score_distributions, score_distributions_cis = rly.create_performance_profile(
                                                            scores_dict, thresholds)

        # Plot.
        fig = plt.figure()
        fig.set_size_inches(FIGURE_X, FIGURE_Y)

        for exp in EXPS_DATA:
            score = score_distributions[exp['label_1']]
            lower_ci, upper_ci = score_distributions_cis[exp['label_1']]
            p = plt.plot(thresholds, score, label=exp['label_1'])
            plt.fill_between(thresholds, lower_ci, upper_ci,
                                color=p[0].get_color(), alpha=0.15)

        plt.xlabel(r'Reward $(\tau)$')
        plt.ylabel(r'Fraction of runs with reward $> \tau$')
        plt.legend()

        plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_performance_profile.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{PLOTS_FOLDER_PATH}/rollout_{rollout_type}_final_performance_profile.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        if STAT_TESTS:
            # Statistical tests for last episode.
            stat_test_data = {}
            for exp in EXPS_DATA:
                rollout_data = rollout_processed_data[exp['id']] # [R,(E)]
                rollout_averaged_per_run = np.average(rollout_data, axis=2) # [R,(E)]
                stat_test_data[exp['id']] = np.mean(rollout_averaged_per_run[:,-10:], axis=1) # [R]

            # Shapiro test (asserts that data is normally distributed for all groups).
            print('Shapiro tests:')
            for exp in EXPS_DATA:
                shapiro_test = stats.shapiro(stat_test_data[exp['id']])
                print(f'\t{exp["label"]}: {shapiro_test}')

            # Parametric Levene test (asserts that all groups share the same variance).
            t_args = [stat_test_data[exp['id']] for exp in EXPS_DATA]
            print(f'\nLevene\'s test: {stats.levene(*t_args)}')

            # Parametric ANOVA test (test whether all groups share the same mean value or not).
            print(f'\nANOVA test: {stats.f_oneway(*t_args)}')

            # Tukey HSD test (pairwise comparisons between all groups).
            t_data, groups = [], []
            for exp in EXPS_DATA:
                groups.extend([exp['label'] for _ in range(len(stat_test_data[exp['id']]))])
                t_data.extend(stat_test_data[exp['id']])
            print('\nTukeyHSD:', pairwise_tukeyhsd(t_data, groups))

            # Non-parametric test.
            print('\nKruskal (non-parametric) test:', stats.kruskal(*t_args))

            # Post-hoc non-parametric comparisons.
            t_data = [stat_test_data[exp['id']] for exp in EXPS_DATA]
            print(sp.posthoc_conover(t_data, p_adjust='holm'))


if __name__ == "__main__":
    main()
