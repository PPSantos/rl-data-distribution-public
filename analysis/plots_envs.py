import os
import pathlib
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/plots/envs'

FIGURE_X = 6.0
FIGURE_Y = 4.0

def main():

    # Prepare plots output folder.
    os.makedirs(PLOTS_FOLDER_PATH, exist_ok=True)

    # Fake colormap.
    cdict1 = {'red':   ((0.0, 0.0, 0.55),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.55),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.55),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
    }
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap('fakeCmap', cdict1)
    cmap.set_bad("black")


    # gridEnv1.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    fake_data = np.zeros((8,8))

    sns.heatmap(fake_data, linewidth=0.5, cmap=cmap, cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-0.30, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, 0.67, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER_PATH}/gridEnv1.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{PLOTS_FOLDER_PATH}/gridEnv1.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # gridEnv4.
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    fake_data = np.zeros((7,8))

    fake_data[3,[1,2,3,4,5,6]] = np.nan

    sns.heatmap(fake_data, linewidth=0.5, cmap=cmap, cbar=False)

    #plt.hlines([0, plt.ylim()[0]], *plt.xlim(), color='black', linewidth=4)
    #plt.vlines([0, plt.xlim()[1]], *plt.ylim(), color='black', linewidth=4)

    plt.text(0.34, plt.ylim()[0]-3.35, 'S', fontsize=16, color='white')
    plt.text(plt.xlim()[1]-0.7, plt.ylim()[0]-3.35, 'G', fontsize=16, color='white')

    plt.xticks([]) # remove the tick marks by setting to an empty list
    plt.yticks([]) # remove the tick marks by setting to an empty list
    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER_PATH}/gridEnv4.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{PLOTS_FOLDER_PATH}/gridEnv4.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main()
