import os
import numpy as np 
import pathlib

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('ggplot')

matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 12})


FIGURE_X = 6.0
FIGURE_Y = 4.0
RED_COLOR = (0.886, 0.29, 0.20)
GRAY_COLOR = (0.2,0.2,0.2)

cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.75, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  1.0, 1.0]]}

SHOW_PLOTS = False
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/'


if __name__ == '__main__':

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH
    os.makedirs(output_folder, exist_ok=True)

    # (Old) 2D plot.
    """ alpha = 1.2
    w_3 = 30

    x_dim = 100
    y_dim = 100
    X = np.linspace(0.0,1.0,x_dim)
    Y = np.linspace(0.0,1.0,y_dim)
    Z = np.zeros((x_dim, y_dim))

    for x, mu_s1_a1 in enumerate(X):
        for y, mu_s2_a1 in enumerate(Y):

            print('mu_s2_a1:', mu_s2_a1)

            w_1 = (mu_s1_a1*100 + alpha*mu_s2_a1*(-35)) / (mu_s1_a1 + alpha**2*mu_s2_a1)
            w_2 = -10 + max(1.2*w_1, w_3)

            correct_actions = 0
            if w_1 > w_2:
                correct_actions += 1
            
            if alpha*w_1 < w_3:
                correct_actions += 1

            Z[x,y] = correct_actions

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # make a color map of fixed colors
    #cmap = matplotlib.colors.ListedColormap(['yellow', 'red', 'blue'])
    cmap = matplotlib.cm.get_cmap('Greys', 3)
    print(cmap)
    bounds=[0,1,2,3]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(Z, origin='lower', cmap=cmap, norm=norm)

    plt.colorbar(cmap=cmap, norm=norm, boundaries=bounds, ticks=[0,1,2,3])


    #cmap = plt.get_cmap('rainbow', np.max(Z)-np.min(Z)+1)
    #mat = plt.matshow(Z,cmap=cmap,vmin = np.min(Z)-.5, vmax = np.max(Z)+.5)
    #plt.colorbar(mat, ticks=np.arange(np.min(Z),np.max(Z)+1))
    #plt.colorbar(cmap='Set3', ticks=np.arange(np.min(Z),np.max(Z)+1))

    plt.xlabel('mu_s2_a1')
    plt.ylabel('mu_s1_a1')
    #plt.legend()

    plt.savefig('{0}/grid.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.savefig('{0}/grid.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0) """

    # code to generate 'correct_actions' plot (oracle version).
    alphas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5] # np.linspace(1.0, 1.0, 1001)
    x_dim = 101
    X = np.linspace(0.0,1.0,x_dim)
    Z = np.zeros((len(alphas), x_dim))        

    w_3 = 30.0

    for a, alpha in enumerate(alphas):
        for x, mu_s1_a1 in enumerate(X):

            mu_s2_a1 = (1 - mu_s1_a1)

            print('mu_s1_a1:', mu_s1_a1)
            print('mu_s2_a1:', mu_s2_a1)

            w_1 = (mu_s1_a1*100.3 + alpha*mu_s2_a1*(-35.0)) / (mu_s1_a1 + alpha*alpha*mu_s2_a1)
            w_2 = 20 # -10 + max(alpha*w_1, w_3)

            correct_actions = 0
            if w_1 > w_2:
                correct_actions += 1

            if alpha*w_1 < w_3:
                correct_actions += 1

            Z[a,x] = correct_actions

            print('correct_actions:', correct_actions)


    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for a, alpha in enumerate(alphas):
        ax.plot(X, Z[a,:], label=r'$\alpha=$ ' + str(alpha))

    ax.set_xlabel('$\mu(s_1,a_1)$')
    ax.set_ylabel('\# correct actions')
    plt.legend(loc=3,ncol=2)

    ax.set_ylim(-0.1,2.1)
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(['0', '1', '2'])
    plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    sec_ax = ax.secondary_xaxis('top')
    sec_ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sec_ax.set_xticklabels(['1.0', '0.8', '0.6', '0.4', '0.2', '0.0'])
    sec_ax.set_xlabel('$\mu(s_2,a_1)$')

    plt.savefig('{0}/plot_1.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_1.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    # averaged (for different alphas) plot.
    """print(X)
    averaged = np.average(Z, axis=0)
    print(averaged)
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    plt.plot(X, averaged)
    plt.xlabel('mus1a1')
    plt.ylabel('Correct actions')
    plt.legend()
    plt.savefig('{0}/2d_averaged_plot.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.savefig('{0}/2d_averaged_plot.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)"""

    # single fixed point calculation.
    """ alpha = 1.2
    mu_s1_a1 = 0.7
    mu_s2_a1 = (1 - mu_s1_a1)
    
    num_iters = 5_000
    w_1 = 0.0
    w_2 = 0.0
    w_3 = 0.0
    eta = 0.01
    for i in range(num_iters):

        w_1 += eta * (mu_s1_a1*(100 + 0.01*max(alpha*w_1, w_3) - w_1) + mu_s2_a1*alpha*(-35 - alpha*w_1))
        w_2 += eta * (-10 + max(alpha*w_1, w_3) - w_2)
        w_3 += eta * (30 - w_3)

    correct_actions = 0
    if w_1 > w_2:
        correct_actions += 1

    if alpha*w_1 < w_3:
        correct_actions += 1

    print('w_1', w_1)
    print('w_2', w_2)
    print('w_3', w_3)
    print('correct_actions:', correct_actions) """

    # Code to generate the one-step TD version (fixed point calculation) correct actions plot.
    alphas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5] # [1.25]
    x_dim = 100
    Y = np.zeros((len(alphas), x_dim))        
    X = np.linspace(0.0, 1.0, x_dim)

    for a, alpha in enumerate(alphas):

        num_iters = 5_000
        w_1 = 0.0
        w_2 = 0.0
        w_3 = 0.0

        #w1_s = []
        #w2_s = []
        #w3_s = []

        eta = 0.01

        for x, mu_s1_a1 in enumerate(X):

            mu_s2_a1 = (1 - mu_s1_a1)

            #print('mu_s1_a1:', mu_s1_a1)
            #print('mu_s2_a1:', mu_s2_a1)

            for i in range(num_iters):

                w_1 += eta * (mu_s1_a1*(100 + 0.01*max(alpha*w_1, w_3) - w_1) + mu_s2_a1*alpha*(-35 - alpha*w_1))
                w_2 += eta * (-10 + max(alpha*w_1, w_3) - w_2)
                w_3 += eta * (30 - w_3)

                #w1_s.append(w_1)
                #w2_s.append(w_2)
                #w3_s.append(w_3)

            print('w_1', w_1)
            print('w_2', w_2)
            print('w_3', w_3)

            correct_actions = 0
            if w_1 > w_2:
                correct_actions += 1

            if alpha*w_1 < w_3:
                correct_actions += 1

            Y[a,x] = correct_actions

            # if x>0 and Y[a,x-1] != Y[a,x]:
            #     print('changeddd')
            #     print(Y[a,x-1])
            #     print(Y[a,x])
            #     print(x)
            #     print(mu_s1_a1)


    #fig = plt.figure()
    #fig.set_size_inches(FIGURE_X, FIGURE_Y)
    #X = np.arange(num_iters)
    #plt.plot(X, w1_s, label='w1')
    #plt.plot(X, w2_s, label='w2')
    #plt.plot(X, w3_s, label='w3')
    #plt.xlabel('Iteration')
    #plt.ylabel('Value')
    #plt.legend()
    #plt.savefig('{0}/iterations.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    # plt.savefig('{0}/iterations.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for a, alpha in enumerate(alphas):
        ax.plot(X, Y[a,:], label=r'$\alpha=$ ' + str(alpha))

    ax.set_xlabel('$\mu(s_1,a_1)$')
    ax.set_ylabel('\# correct actions')
    plt.legend(loc=3,ncol=2)

    ax.set_ylim(-0.1,2.1)
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(['0', '1', '2'])
    plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    sec_ax = ax.secondary_xaxis('top')
    sec_ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sec_ax.set_xticklabels(['1.0', '0.8', '0.6', '0.4', '0.2', '0.0'])
    sec_ax.set_xlabel('$\mu(s_2,a_1)$')

    plt.savefig('{0}/plot_2.png'.format(output_folder), bbox_inches='tight', pad_inches=0)
    plt.savefig('{0}/plot_2.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
