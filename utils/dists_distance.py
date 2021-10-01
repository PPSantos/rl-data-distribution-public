import sys
import os
from tqdm import tqdm
import json
import time
import numpy as np
import pathlib
from datetime import datetime
import scipy

from rlutil.json_utils import NumpyEncoder
from envs import env_suite

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

FIGURE_X = 8.0
FIGURE_Y = 6.0


DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'


ARGS = {
    'd_1': '/home/ppsantos/GAIPS/RL_playground/data/gridEnv1_sampling_dist_2021-10-01-01-05-42/data.json',
    'd_2': '/home/ppsantos/GAIPS/RL_playground/data/gridEnv1_sampling_dist_2021-10-01-01-09-27/data.json',
}

def main():

    # Load distribution 1.
    with open(ARGS['d_1'], 'r') as f:
        data = json.load(f)
        d_1 = np.array(json.loads(data)['sampling_dist'])
    f.close()

    # Load distribution 2.
    with open(ARGS['d_2'], 'r') as f:
        data = json.load(f)
        d_2 = np.array(json.loads(data)['sampling_dist'])
    f.close()

    #print('d_1', d_1)
    #print('d_2', d_2)

    wass_dist = scipy.stats.wasserstein_distance(d_1, d_2)

    print('wass_dist', wass_dist)

    ratio_dist = np.max(d_1/(d_2+1e-06))
    print('ratio_dist', ratio_dist)

    kl_div = scipy.stats.entropy(d_1,d_2)
    print('KL div.', kl_div)

if __name__ == "__main__":
    main()
