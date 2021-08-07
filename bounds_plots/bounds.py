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

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams['text.usetex'] = True


FIGURE_X = 6.0
FIGURE_Y = 4.0
RED_COLOR = (0.886, 0.29, 0.20)
GRAY_COLOR = (0.2,0.2,0.2)

SHOW_PLOTS = False
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/bounds_plots/plots/'

args = {
    'num_states': 10,
    'mdp_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'mu_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_mdps': 50,
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
    Ps = {}
    for mdp_alpha in args['mdp_dirichlet_alphas']:

        Ps_aux = np.zeros((args['num_mdps'], args['num_states'], args['num_states']))
        for i in range(args['num_mdps']):

            P = np.zeros((args['num_states'],args['num_states']))
            for state in range(args['num_states']):
                P[state,:] = np.random.dirichlet([mdp_alpha]*args['num_states'])

            Ps_aux[i,:,:] = P

        Ps[mdp_alpha] = Ps_aux

    to_plot = {}
    for mdp_alpha in args['mdp_dirichlet_alphas']:

        Cs = []
        for mu_alpha in args['mu_dirichlet_alphas']:

            print('mu_alpha=', mu_alpha)
            c_aux = []

            for _ in range(args['num_trials']):
                mu = np.random.dirichlet([mu_alpha]*args['num_states'])

                for p in Ps[mdp_alpha]:

                    # Calculate bound C value.
                    max_c = np.max(np.divide(p[0,:],(mu+1e-04)))
                    for j in range(args['num_states'])[1:]:
                        max_c = max(max_c, np.max(np.divide(p[j,:],(mu+1e-04))))

                    c_aux.append(max_c)

            Cs.append(c_aux)

        to_plot[mdp_alpha] = np.mean(np.array(Cs), axis=1)


    """
        X Axis = Alpha (Dirichlet parameter)
    """
    # Linear y-scale.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for mdp_alpha in args['mdp_dirichlet_alphas']:

        plt.scatter(args['mu_dirichlet_alphas'], to_plot[mdp_alpha])
        plt.plot(args['mu_dirichlet_alphas'], to_plot[mdp_alpha], label=f'MDP alpha={mdp_alpha}')
    
    plt.xlabel('Alpha (Dirichlet parameter)')
    plt.ylabel('Average C value')
    plt.legend()

    plt.savefig('{0}/plot_1_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_1_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale.
    plt.yscale("log")
    plt.savefig('{0}/plot_1_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_1_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale (averaged).
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    aux = np.array([to_plot[k] for k in sorted(to_plot.keys())])
    print('aux shapeeeeeeeeee', aux.shape)
    averaged = np.mean(aux, axis=0)
    stds = np.std(aux, axis=0)

    plt.scatter(args['mu_dirichlet_alphas'], averaged)
    plt.plot(args['mu_dirichlet_alphas'], averaged)
    
    plt.xlabel('Alpha (Dirichlet parameter)')
    plt.ylabel('Average C value')
    plt.yscale("log")

    plt.savefig('{0}/plot_1_3.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_1_3.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)


    """
        X Axis = Sampling dist. entropy
    """
    # Linear y-scale.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for mdp_alpha in args['mdp_dirichlet_alphas']:

        plt.scatter(entropies, to_plot[mdp_alpha])
        plt.plot(entropies, to_plot[mdp_alpha], label=f'MDP alpha={mdp_alpha}')
    
    plt.xlabel('Sampling dist. entropy')
    plt.ylabel('Average C value')
    plt.legend()

    plt.savefig('{0}/plot_2_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale.
    plt.yscale("log")
    plt.savefig('{0}/plot_2_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Linear y-scale (averaged).
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.scatter(entropies, averaged)
    p = plt.plot(entropies, averaged, label='Mean')

    plt.fill_between(entropies, averaged-stds, averaged+stds, color=p[0].get_color(), alpha=0.3, label='Std')

    plt.xlabel(r'$\mathcal{H}(\mu)$')
    plt.ylabel(r'$C_1$')

    plt.legend()

    plt.savefig('{0}/plot_2_3.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_3.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale (averaged).
    plt.yscale("log")
    plt.savefig('{0}/plot_2_4.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_4.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)


    ####################################################################################################
    # Linear y-scale (averaged).
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    p_50 = np.percentile(aux, 50, axis=0)

    p_75 = np.percentile(aux, 75, axis=0)
    p_25 = np.percentile(aux, 25, axis=0)

    p_37_5 = np.percentile(aux, 37.5, axis=0)
    p_62_5 = np.percentile(aux, 62.5, axis=0)

    p_12_5 = np.percentile(aux, 12.5, axis=0)
    p_87_5 = np.percentile(aux, 87.5, axis=0)

    plt.fill_between(entropies, p_37_5, p_62_5, color=RED_COLOR, alpha=0.6, label='Pct25')
    plt.fill_between(entropies, p_25, p_75, color=RED_COLOR, alpha=0.25, label='Pct50')
    plt.fill_between(entropies, p_12_5, p_87_5, color=RED_COLOR, alpha=0.1, label='Pct75')

    plt.scatter(entropies, p_50, color=GRAY_COLOR)
    p = plt.plot(entropies, p_50, label='Median', color=GRAY_COLOR)

    plt.xlabel(r'$\mathcal{H}(\mu)$')
    plt.ylabel(r'$C_1$')

    plt.legend()

    plt.savefig('{0}/plot_perc_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_perc_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale (averaged).
    plt.yscale("log")
    plt.savefig('{0}/plot_perc_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_perc_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
