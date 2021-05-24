import os
import json
import numpy as np 
import tensorflow as tf
import scipy
from scipy.spatial import distance
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from rlutil.json_utils import NumpyEncoder


FIGURE_X = 6.0
FIGURE_Y = 4.0


SHOW_PLOTS = False
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/analysis/plots/'

args = {
    'num_states': 5,
    'number_of_functions': 10,
    'dirichlet_alphas': [0.05], #, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    'num_dirichlet_samples': 5,
    'resample_size': 10_000,
}



if __name__ == '__main__':

    # Prepare plots output folder.
    output_folder = PLOTS_FOLDER_PATH + exp_id + '/'
    os.makedirs(output_folder, exist_ok=True)

    for alpha in args['dirichlet_alphas']:

        for _ in range(args['num_dirichlet_samples']):

            P = np.array([]) # TODO




        
            # Resample data.
            X_resampled = []
            Y_resampled = []
            dirichlet_sample = np.random.dirichlet([alpha]*args['nb_of_samples'])
            for _ in range(args['resample_size']):
                idx = np.random.choice(range(len(X)), p=dirichlet_sample)
                X_resampled.append(X[idx])
                Y_resampled.append(Y[idx])
            X_resampled = np.array(X_resampled)
            Y_resampled = np.array(Y_resampled)

            # Plot resampled data.
            if SHOW_PLOTS:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(FIGURE_X, FIGURE_Y)
                ax1.scatter(X_resampled, Y_resampled, color='blue', label='Resampled data')
                ax1.legend()
                ax2 = ax1.twinx()
                ax2.hist(X_resampled, 100, density=True, label='p(x)')
                ax2.set_ylim([0,1.05])
                plt.title('Resampled data')
                plt.legend()
