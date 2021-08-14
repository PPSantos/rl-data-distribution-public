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

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# matplotlib.rcParams['text.usetex'] = True


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
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/bounds_plots/mdp1/'


if __name__ == '__main__':

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH
    os.makedirs(output_folder, exist_ok=True)

    alpha = 1.2
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
    # plt.savefig('{0}/grid.pdf'.format(output_folder), bbox_inches='tight', pad_inches=0)
