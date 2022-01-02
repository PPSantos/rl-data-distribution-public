import os
import json
import numpy as np 
import scipy
from scipy.spatial import distance
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('ggplot')

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 14})

FIGURE_X = 6.0
FIGURE_Y = 4.0

RED_COLOR = (0.886, 0.29, 0.20)
GRAY_COLOR = (0.2,0.2,0.2)

PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/theoretical_bounds/plots'
print(PLOTS_FOLDER_PATH)

args = {
    'num_states': 100,
    'dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0],
    'num_dists': 50, # 50 for the coverage plots, 100 for the distances plots.
    'dataset_sizes': [1_000, 5_000, 10_000, 20_000],
}

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

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH
    os.makedirs(output_folder, exist_ok=True)

    entropies = []
    sampled_entropies = []
    dists = {}
    for alpha in sorted(args['dirichlet_alphas']):
        expected_entropy = scipy.special.digamma(args['num_states']*alpha + 1) - scipy.special.digamma(alpha + 1)
        entropies.append(expected_entropy)

        samples = np.zeros(1_000)
        for s in range(1_000):
            dist = np.random.dirichlet([alpha]*args['num_states'])
            samples[s] = scipy.stats.entropy(dist)

        sampled_entropies.append(np.mean(samples))

        dists[alpha] = [np.random.dirichlet([alpha]*args['num_states'])
                        for _ in range(args['num_dists'])]

    # Dataset coverage.
    # X: expected entropy; Y: dataset coverage.
    covs = []
    for d_size in args['dataset_sizes']:
        print('d_size=', d_size)
        
        covs_dataset = {}
        for alpha in sorted(args['dirichlet_alphas']):

            covs_dists = []
            for dist in dists[alpha]:
                # Create a dataset by sampling from the distribution.
                dataset_counts = np.zeros(args['num_states'])
                for _ in range(d_size):
                    sampled_state = np.random.choice(np.arange(args['num_states']), p=dist)
                    dataset_counts[sampled_state] += 1
                covs_dists.append(np.sum(dataset_counts>0))

            covs_dataset[alpha] = covs_dists

        covs.append(covs_dataset)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for (d_size, data) in zip(args['dataset_sizes'], covs):

        to_plot_data = np.array([np.mean(data[alpha]) / args['num_states']
                            for alpha in sorted(args['dirichlet_alphas'])])
        print('to_plot_data', to_plot_data)

        p = plt.plot(entropies, to_plot_data, label=r'size($\mathcal{D}$)=' + str(d_size))
        plt.scatter(entropies, to_plot_data, color=p[0].get_color())

    plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')
    plt.ylabel('Coverage (\%)')

    plt.legend(loc=4)

    plt.yscale("linear")
    plt.savefig('{0}/dataset_coverage.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/dataset_coverage.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Distance to all other distributions.
    # X: expected entropy; Y: distance to all other distributions (chi-square div.).
    """ def f_div(x, y):
        y = y + 1e-06
        # return np.dot(y, ((x/y)-1)**2 )
        return np.dot(y, (x/y)**2 - 1)

    all_dists = []
    for alpha in sorted(args['dirichlet_alphas']):
        all_dists.extend(dists[alpha])

    print(len(all_dists))

    distances = {}
    for alpha in sorted(args['dirichlet_alphas']):

        ds = []
        for mu_dist in dists[alpha]:
            # Calculate mean distance to all_dists.
            ds.append(np.mean([f_div(other_dist, mu_dist) for other_dist in all_dists]))

        distances[alpha] = ds

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    point_estimations = []
    lower_ci_bounds = []
    upper_ci_bounds = []
    for alpha in sorted(args['dirichlet_alphas']):
        point_est, (lower_ci, upper_ci) = mean_agg_func(distances[alpha])
        point_estimations.append(point_est)
        lower_ci_bounds.append(lower_ci)
        upper_ci_bounds.append(upper_ci)

    p = plt.plot(entropies, point_estimations, color=GRAY_COLOR, label='Mean', zorder=10)
    plt.scatter(entropies, point_estimations, color=GRAY_COLOR, zorder=10)
    plt.fill_between(entropies, lower_ci_bounds, upper_ci_bounds,
            color=p[0].get_color(), alpha=0.15, label='95\% mean C.I.')

    plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')
    plt.ylabel(r'$\chi^2(\cdot||\mu)$')

    plt.legend()

    plt.yscale("log")
    plt.savefig('{0}/distance_to_all_dists.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/distance_to_all_dists.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close() """

    # Distance to all other distributions.
    # X: expected entropy; Y: distance to all other distributions (KL div.).
    """ all_dists = []
    for alpha in sorted(args['dirichlet_alphas']):
        all_dists.extend(dists[alpha])

    print(len(all_dists))

    distances = {}
    for alpha in sorted(args['dirichlet_alphas']):

        ds = []
        for mu_dist in dists[alpha]:
            # Calculate mean distance to all_dists.
            ds.append(np.mean([scipy.stats.entropy(other_dist, mu_dist) for other_dist in all_dists]))

        distances[alpha] = ds

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    point_estimations = []
    lower_ci_bounds = []
    upper_ci_bounds = []
    for alpha in sorted(args['dirichlet_alphas']):
        point_est, (lower_ci, upper_ci) = mean_agg_func(distances[alpha])
        point_estimations.append(point_est)
        lower_ci_bounds.append(lower_ci)
        upper_ci_bounds.append(upper_ci)

    p = plt.plot(entropies, point_estimations, color=GRAY_COLOR, label='Mean', zorder=10)
    plt.scatter(entropies, point_estimations, color=GRAY_COLOR, zorder=10)
    plt.fill_between(entropies, lower_ci_bounds, upper_ci_bounds,
            color=p[0].get_color(), alpha=0.15, label='95\% mean C.I.')

    plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')
    plt.ylabel(r'KL$(\cdot||\mu)$')

    plt.legend()

    plt.yscale("linear")
    plt.savefig('{0}/distance_to_all_dists_KL.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/distance_to_all_dists_KL.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.close() """

if __name__ == '__main__':
    main()