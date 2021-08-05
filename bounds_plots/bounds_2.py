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


SHOW_PLOTS = False
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/bounds_plots/plots_2/'

args = {

    'gamma': 0.9,

    # P^pi matrices args.
    'num_states': 10,
    'mdp_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_mdps': 50,

    # initial states distributions.
    'init_dists_dirichlet_alphas': [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
    'num_init_dists': 20,

    # mu dists args.
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

        Cs = []
        for trial in range(args['num_trials']):

            # Randomly select initial state dist.
            init_dist = init_dists[np.random.choice(init_dists.shape[0])] # [S]
            
            # Randomly select mu dist.
            mu_dist = np.random.dirichlet([mu_alpha]*args['num_states']) # [S]

            # Calculate c(m) for m=1.
            P = Ps[np.random.choice(init_dists.shape[0])] # [S,S]
            Ps_product = np.dot(init_dist,P) # [S]

            c_1 = np.max(Ps_product / (mu_dist + 1e-04))
            #cms = [c_1]
            C_value = c_1

            for m in range(2,50): # number of MDP timesteps.
                
                # Randomly select P^pi matrix.
                P = Ps[np.random.choice(init_dists.shape[0])]
                Ps_product = np.dot(Ps_product,P)

                c_m = np.max(Ps_product / (mu_dist + 1e-04))
                #cms.append(c_m)

                C_value += m * c_m * args['gamma']**(m-1)

            # Calculate bound C value.
            C_value = (1-args['gamma'])**2 * C_value

            print('C_val', C_value)

            Cs.append(C_value)

        to_plot[mu_alpha] = Cs

    print(to_plot)

    exit()

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
