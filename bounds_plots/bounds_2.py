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


SHOW_PLOTS = False
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/bounds_plots/plots_2/'

args = {

    # MDP args.
    'gamma': 0.9,
    'num_states': 10,
    'mdp_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_mdps': 50,

    # initial states distributions.
    'init_dists_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_init_dists': 20,

    # mu dists args.
    'mu_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_trials': 10_000, # TODO: increae
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
    for alpha in sorted(args['mu_dirichlet_alphas']):
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
    # Generate a set of P^pi matrices.
    Ps = []
    for mdp_alpha in args['mdp_dirichlet_alphas']:
        for j in range(args['num_mdps']):
            P = np.zeros((args['num_states'],args['num_states']))
            for state in range(args['num_states']):
                P[state,:] = np.random.dirichlet([mdp_alpha]*args['num_states'])
            Ps.append(P)
    Ps = np.array(Ps)
    print('Ps', Ps.shape)

    # Generate a set of initial dists.
    init_dists = []
    for init_dist_alpha in args['init_dists_dirichlet_alphas']:
        for j in range(args['num_init_dists']):
            init_dists.append(np.random.dirichlet([init_dist_alpha]*args['num_states']))
    init_dists = np.array(init_dists)
    print('init_dists', init_dists.shape)

    to_plot = {}
    for mu_alpha in args['mu_dirichlet_alphas']:
        print('mu_alpha=', mu_alpha)

        Cs = []
        for trial in range(args['num_trials']):

            # Randomly select initial state dist.
            init_dist = init_dists[np.random.choice(init_dists.shape[0])] # [S]
            
            # Randomly select mu dist.
            mu_dist = np.random.dirichlet([mu_alpha]*args['num_states']) # [S]

            # Calculate c(m) for m=1.
            P = Ps[np.random.choice(Ps.shape[0])] # [S,S]
            Ps_product = np.dot(init_dist,P) # [S]

            #C_value = np.max(Ps_product / (mu_dist + 1e-04))
            C_value = np.sqrt(np.dot(mu_dist, (Ps_product / (mu_dist + 1e-04))**2))
            for m in range(2,50): # number of MDP timesteps.
                
                # Randomly select P^pi matrix.
                P = Ps[np.random.choice(Ps.shape[0])]
                Ps_product = np.dot(Ps_product,P)

                #c_m = np.max(Ps_product / (mu_dist + 1e-04))
                c_m = np.sqrt(np.dot(mu_dist, (Ps_product / (mu_dist + 1e-04))**2))

                C_value += m * c_m * args['gamma']**(m-1)

            # Calculate bound C value.
            C_value = (1-args['gamma'])**2 * C_value
            Cs.append(C_value)

        to_plot[mu_alpha] = Cs

    aux = np.array([to_plot[k] for k in sorted(to_plot.keys())])
    print('aux', aux.shape)

    """ # Linear y-scale (averaged).
    averaged = np.mean(aux, axis=1)
    stds = np.std(aux, axis=1)
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    plt.scatter(entropies, averaged)
    p = plt.plot(entropies, averaged, label='Mean')

    print(averaged-stds)
    print(averaged+stds)

    plt.fill_between(entropies, averaged-stds, averaged+stds, color=p[0].get_color(), alpha=0.3, label='Std')

    plt.xlabel(r'$\mathcal{H}(\mu)$')
    plt.ylabel(r'$C_2$')

    plt.legend()

    #plt.savefig('{0}/plot_2_3.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    #plt.savefig('{0}/plot_2_3.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale (averaged).
    plt.yscale("log")
    plt.savefig('{0}/plot_2_4.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2_4.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0) """

    ####################################################################################################
    # Linear y-scale (averaged).
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    p_50 = np.percentile(aux, 50, axis=1)

    p_75 = np.percentile(aux, 75, axis=1)
    p_25 = np.percentile(aux, 25, axis=1)

    p_37_5 = np.percentile(aux, 37.5, axis=1)
    p_62_5 = np.percentile(aux, 62.5, axis=1)

    p_12_5 = np.percentile(aux, 12.5, axis=1)
    p_87_5 = np.percentile(aux, 87.5, axis=1)

    plt.fill_between(entropies, p_37_5, p_62_5, color=RED_COLOR, alpha=0.6, label='25th pct.')
    plt.fill_between(entropies, p_25, p_75, color=RED_COLOR, alpha=0.25, label='50th pct.')
    plt.fill_between(entropies, p_12_5, p_87_5, color=RED_COLOR, alpha=0.1, label='75th pct.')

    plt.scatter(entropies, p_50, color=GRAY_COLOR)
    p = plt.plot(entropies, p_50, label='Median', color=GRAY_COLOR)

    plt.xlabel(r'$\mathcal{H}(\mu)$')
    plt.ylabel(r'$C_2$')

    plt.legend(loc=3)

    plt.savefig('{0}/plot_perc_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_perc_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # Log y-scale (averaged).
    plt.yscale("log")
    plt.savefig('{0}/plot_perc_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_perc_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
