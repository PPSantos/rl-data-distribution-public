import os
import json
import numpy as np 
import scipy
from scipy.spatial import distance
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from utils.json_utils import NumpyEncoder


FIGURE_X = 6.0
FIGURE_Y = 4.0


SHOW_PLOTS = False
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/bounds_plots/plots/plots_2/'

args = {
    'num_states': 4,
    'mu_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_trials': 50,
}


if __name__ == '__main__':

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH
    os.makedirs(output_folder, exist_ok=True)

    # Entropy plot.
    entropies = []
    sampled_entropies = []
    coverages_1 = []
    coverages_2 = []
    for alpha in args['mu_dirichlet_alphas']:
        expected_entropy = scipy.special.digamma(args['num_states']*alpha + 1) - scipy.special.digamma(alpha + 1)
        entropies.append(expected_entropy)

        samples = np.zeros(1_000)
        covs_1 = np.zeros(1_000)
        covs_2 = np.zeros(1_000)
        for s in range(1_000):
            dist = np.random.dirichlet([alpha]*args['num_states'])
            samples[s] = scipy.stats.entropy(dist)
            covs_1[s] = np.sum(dist > 1e-02)
            covs_2[s] = np.sum(dist > 1e-03)

        sampled_entropies.append(np.mean(samples))
        coverages_1.append(np.mean(covs_1))
        coverages_2.append(np.mean(covs_2))

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(args['mu_dirichlet_alphas'], entropies, label='Expected entropy')
    plt.plot(args['mu_dirichlet_alphas'], sampled_entropies, label='Sampled entropy')
    plt.xlabel('Alpha (Dirichlet parameter)')
    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('{0}/entropy.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/entropy.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Coverage plot.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(args['mu_dirichlet_alphas'], np.array(coverages_1) / args['num_states'], label='1e-02 threshold')
    plt.plot(args['mu_dirichlet_alphas'], np.array(coverages_2) / args['num_states'], label='1e-03 threshold')
    plt.xlabel('Alpha (Dirichlet parameter)')
    plt.ylabel('Coverage')
    plt.legend()
    plt.savefig('{0}/coverage_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/coverage_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(entropies, np.array(coverages_1) / args['num_states'], label='1e-02 threshold')
    plt.plot(entropies, np.array(coverages_2) / args['num_states'], label='1e-03 threshold')
    plt.xlabel('Entropy')
    plt.ylabel('Coverage')
    plt.legend()
    plt.savefig('{0}/coverage_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/coverage_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    print('Calculating bounds values...')
    # Generate a set of MDPs (transition probs.).

    P = np.array([  [[0.0, 0.01, 0.99, 0.0], [0.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.0,  0.0,  1.0], [0.0, 0.0, 1.0, 0.0]],
                    [[0.0, 0.0,  1.0,  0.0], [0.0, 0.0, 1.0, 0.0]],
                    [[0.0, 0.0,  0.0,  1.0], [0.0, 0.0, 0.0, 1.0]],
                ])

    # action_range = np.linspace(0,1.0,11)
    # mesh = np.array(np.meshgrid(action_range, action_range))
    # combinations = mesh.T.reshape(-1, 2)
    action_combinations = [[1.0,0.0],[0.9,0.1],[0.8,0.2],[0.7,0.3],
                    [0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7],
                    [0.2,0.8],[0.1,0.9],[0.0,1.0]]

    Ps = [action[0]*P[:,0,:] + action[1]*P[:,1,:] for action in action_combinations]

    Cs = []
    for mu_alpha in args['mu_dirichlet_alphas']:

        print('mu_alpha=', mu_alpha)
        c_aux = []

        for _ in range(args['num_trials']):
            mu = np.random.dirichlet([mu_alpha]*args['num_states'])

            for p in Ps:

                # Calculate bound C value.
                max_c = np.max(np.divide(p[0,:],(mu+1e-04)))
                for j in range(args['num_states'])[1:]:
                    max_c = max(max_c, np.max(np.divide(p[j,:],(mu+1e-04))))

                c_aux.append(max_c)

        Cs.append(c_aux)

    result = np.mean(np.array(Cs), axis=1)

    print(result)

    """
        X Axis = Alpha (Dirichlet parameter)
    """
    # Linear y-scale.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.scatter(args['mu_dirichlet_alphas'], result)
    plt.plot(args['mu_dirichlet_alphas'], result)
    
    plt.xlabel('Alpha (Dirichlet parameter)')
    plt.ylabel('Average C value')

    plt.savefig('{0}/plot_1_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_1_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale.
    plt.yscale("log")
    plt.savefig('{0}/plot_1_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_1_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    """
        X Axis = Sampling dist. entropy
    """
    # Linear y-scale.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.scatter(entropies, result)
    plt.plot(entropies, result)
    
    plt.xlabel('Sampling dist. entropy')
    plt.ylabel('Average C value')
    plt.legend()

    plt.savefig('{0}/plot_2_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale.
    plt.yscale("log")
    plt.savefig('{0}/plot_2_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)