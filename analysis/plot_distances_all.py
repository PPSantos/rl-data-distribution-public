import os
import json
import numpy as np
import pathlib
from datetime import datetime
import scipy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

from statsmodels.nonparametric.smoothers_lowess import lowess

#################################################################
DATA = {
    'gridEnv1': {
        'OPTIMAL_SAMPLING_DIST_IDS': [
        'gridEnv1_sampling_dist_2021-10-06-10-48-59', 'gridEnv1_sampling_dist_2021-10-06-10-49-04', 'gridEnv1_sampling_dist_2021-10-06-10-49-08', 'gridEnv1_sampling_dist_2021-10-06-10-49-12', 'gridEnv1_sampling_dist_2021-10-06-10-49-16', 'gridEnv1_sampling_dist_2021-10-06-10-49-21', 'gridEnv1_sampling_dist_2021-10-06-10-49-25', 'gridEnv1_sampling_dist_2021-10-06-10-49-29', 'gridEnv1_sampling_dist_2021-10-06-10-49-33', 'gridEnv1_sampling_dist_2021-10-06-10-49-37', 'gridEnv1_sampling_dist_2021-10-06-10-49-42', 'gridEnv1_sampling_dist_2021-10-06-10-49-46', 'gridEnv1_sampling_dist_2021-10-06-10-49-50', 'gridEnv1_sampling_dist_2021-10-06-10-49-54', 'gridEnv1_sampling_dist_2021-10-06-10-49-58', 'gridEnv1_sampling_dist_2021-10-06-10-50-03', 'gridEnv1_sampling_dist_2021-10-06-10-50-07', 'gridEnv1_sampling_dist_2021-10-06-10-50-11', 'gridEnv1_sampling_dist_2021-10-06-10-50-15', 'gridEnv1_sampling_dist_2021-10-06-10-50-19',
        ],
        'OFFLINE_DQN_EXP_IDS': [
        'gridEnv1_offline_dqn_2021-10-06-00-38-26', 'gridEnv1_offline_dqn_2021-10-06-00-49-56', 'gridEnv1_offline_dqn_2021-10-06-01-01-29', 'gridEnv1_offline_dqn_2021-10-06-01-12-58', 'gridEnv1_offline_dqn_2021-10-06-01-24-35', 'gridEnv1_offline_dqn_2021-10-06-01-36-06', 'gridEnv1_offline_dqn_2021-10-06-01-47-40', 'gridEnv1_offline_dqn_2021-10-06-01-59-08', 'gridEnv1_offline_dqn_2021-10-06-02-10-43', 'gridEnv1_offline_dqn_2021-10-06-02-22-23', 'gridEnv1_offline_dqn_2021-10-06-02-33-56', 'gridEnv1_offline_dqn_2021-10-06-02-45-25', 'gridEnv1_offline_dqn_2021-10-06-02-57-00', 'gridEnv1_offline_dqn_2021-10-06-03-08-31', 'gridEnv1_offline_dqn_2021-10-06-03-20-06', 'gridEnv1_offline_dqn_2021-10-06-03-31-41', 'gridEnv1_offline_dqn_2021-10-06-03-43-14', 'gridEnv1_offline_dqn_2021-10-06-03-54-53', 'gridEnv1_offline_dqn_2021-10-06-04-06-33', 'gridEnv1_offline_dqn_2021-10-06-04-18-05', 'gridEnv1_offline_dqn_2021-10-06-04-29-41', 'gridEnv1_offline_dqn_2021-10-06-04-41-12', 'gridEnv1_offline_dqn_2021-10-06-04-52-50', 'gridEnv1_offline_dqn_2021-10-06-05-04-23', 'gridEnv1_offline_dqn_2021-10-06-05-15-57', 'gridEnv1_offline_dqn_2021-10-06-05-27-23', 'gridEnv1_offline_dqn_2021-10-06-05-39-03', 'gridEnv1_offline_dqn_2021-10-06-05-50-39', 'gridEnv1_offline_dqn_2021-10-06-06-02-17', 'gridEnv1_offline_dqn_2021-10-06-06-13-50', 'gridEnv1_offline_dqn_2021-10-06-06-25-26', 'gridEnv1_offline_dqn_2021-10-06-06-37-00', 'gridEnv1_offline_dqn_2021-10-06-06-48-31', 'gridEnv1_offline_dqn_2021-10-06-07-00-03', 'gridEnv1_offline_dqn_2021-10-06-07-11-38', 'gridEnv1_offline_dqn_2021-10-06-07-23-15', 'gridEnv1_offline_dqn_2021-10-06-07-34-47', 'gridEnv1_offline_dqn_2021-10-06-07-46-25', 'gridEnv1_offline_dqn_2021-10-06-07-57-58', 'gridEnv1_offline_dqn_2021-10-06-08-09-29', 'gridEnv1_offline_dqn_2021-10-06-08-21-00', 'gridEnv1_offline_dqn_2021-10-06-08-32-35', 'gridEnv1_offline_dqn_2021-10-06-08-44-05', 'gridEnv1_offline_dqn_2021-10-06-08-55-36', 'gridEnv1_offline_dqn_2021-10-06-09-07-15', 'gridEnv1_offline_dqn_2021-10-06-09-22-06', 'gridEnv1_offline_dqn_2021-10-06-09-33-38', 'gridEnv1_offline_dqn_2021-10-06-09-45-42', 'gridEnv1_offline_dqn_2021-10-06-10-02-25', 'gridEnv1_offline_dqn_2021-10-06-10-19-23',
        'gridEnv1_offline_dqn_2021-10-06-11-05-08', 'gridEnv1_offline_dqn_2021-10-06-11-19-51', 'gridEnv1_offline_dqn_2021-10-06-11-32-15', 'gridEnv1_offline_dqn_2021-10-06-11-44-52', 'gridEnv1_offline_dqn_2021-10-06-11-58-43', 'gridEnv1_offline_dqn_2021-10-06-12-17-10', 'gridEnv1_offline_dqn_2021-10-06-12-34-48', 'gridEnv1_offline_dqn_2021-10-06-12-51-08', 'gridEnv1_offline_dqn_2021-10-06-13-03-10', 'gridEnv1_offline_dqn_2021-10-06-13-15-24', 'gridEnv1_offline_dqn_2021-10-06-13-27-09', 'gridEnv1_offline_dqn_2021-10-06-13-50-43', 'gridEnv1_offline_dqn_2021-10-06-14-07-09', 'gridEnv1_offline_dqn_2021-10-06-14-23-56', 'gridEnv1_offline_dqn_2021-10-06-14-41-14', 'gridEnv1_offline_dqn_2021-10-06-14-53-56', 'gridEnv1_offline_dqn_2021-10-06-15-08-50', 'gridEnv1_offline_dqn_2021-10-06-15-28-32', 'gridEnv1_offline_dqn_2021-10-06-15-45-58', 'gridEnv1_offline_dqn_2021-10-06-16-02-16', 'gridEnv1_offline_dqn_2021-10-06-16-23-42', 'gridEnv1_offline_dqn_2021-10-06-16-46-56', 'gridEnv1_offline_dqn_2021-10-06-17-06-12', 'gridEnv1_offline_dqn_2021-10-06-17-22-53', 'gridEnv1_offline_dqn_2021-10-06-17-41-38',
        ],
        'SAMPLING_DISTS_IDS': [
        'gridEnv1_sampling_dist_2021-10-06-00-38-16', 'gridEnv1_sampling_dist_2021-10-06-00-49-47', 'gridEnv1_sampling_dist_2021-10-06-01-01-20', 'gridEnv1_sampling_dist_2021-10-06-01-12-50', 'gridEnv1_sampling_dist_2021-10-06-01-24-26', 'gridEnv1_sampling_dist_2021-10-06-01-35-57', 'gridEnv1_sampling_dist_2021-10-06-01-47-31', 'gridEnv1_sampling_dist_2021-10-06-01-58-59', 'gridEnv1_sampling_dist_2021-10-06-02-10-34', 'gridEnv1_sampling_dist_2021-10-06-02-22-14', 'gridEnv1_sampling_dist_2021-10-06-02-33-47', 'gridEnv1_sampling_dist_2021-10-06-02-45-16', 'gridEnv1_sampling_dist_2021-10-06-02-56-51', 'gridEnv1_sampling_dist_2021-10-06-03-08-22', 'gridEnv1_sampling_dist_2021-10-06-03-19-57', 'gridEnv1_sampling_dist_2021-10-06-03-31-33', 'gridEnv1_sampling_dist_2021-10-06-03-43-05', 'gridEnv1_sampling_dist_2021-10-06-03-54-44', 'gridEnv1_sampling_dist_2021-10-06-04-06-25', 'gridEnv1_sampling_dist_2021-10-06-04-17-56', 'gridEnv1_sampling_dist_2021-10-06-04-29-33', 'gridEnv1_sampling_dist_2021-10-06-04-41-03', 'gridEnv1_sampling_dist_2021-10-06-04-52-41', 'gridEnv1_sampling_dist_2021-10-06-05-04-14', 'gridEnv1_sampling_dist_2021-10-06-05-15-48', 'gridEnv1_sampling_dist_2021-10-06-05-27-14', 'gridEnv1_sampling_dist_2021-10-06-05-38-54', 'gridEnv1_sampling_dist_2021-10-06-05-50-30', 'gridEnv1_sampling_dist_2021-10-06-06-02-08', 'gridEnv1_sampling_dist_2021-10-06-06-13-41', 'gridEnv1_sampling_dist_2021-10-06-06-25-17', 'gridEnv1_sampling_dist_2021-10-06-06-36-51', 'gridEnv1_sampling_dist_2021-10-06-06-48-22', 'gridEnv1_sampling_dist_2021-10-06-06-59-55', 'gridEnv1_sampling_dist_2021-10-06-07-11-29', 'gridEnv1_sampling_dist_2021-10-06-07-23-06', 'gridEnv1_sampling_dist_2021-10-06-07-34-39', 'gridEnv1_sampling_dist_2021-10-06-07-46-17', 'gridEnv1_sampling_dist_2021-10-06-07-57-49', 'gridEnv1_sampling_dist_2021-10-06-08-09-20', 'gridEnv1_sampling_dist_2021-10-06-08-20-52', 'gridEnv1_sampling_dist_2021-10-06-08-32-26', 'gridEnv1_sampling_dist_2021-10-06-08-43-57', 'gridEnv1_sampling_dist_2021-10-06-08-55-28', 'gridEnv1_sampling_dist_2021-10-06-09-07-06', 'gridEnv1_sampling_dist_2021-10-06-09-21-57', 'gridEnv1_sampling_dist_2021-10-06-09-33-29', 'gridEnv1_sampling_dist_2021-10-06-09-45-29', 'gridEnv1_sampling_dist_2021-10-06-10-02-12', 'gridEnv1_sampling_dist_2021-10-06-10-19-10',
        'gridEnv1_sampling_dist_2021-10-06-11-04-56', 'gridEnv1_sampling_dist_2021-10-06-11-19-42', 'gridEnv1_sampling_dist_2021-10-06-11-32-06', 'gridEnv1_sampling_dist_2021-10-06-11-44-44', 'gridEnv1_sampling_dist_2021-10-06-11-58-30', 'gridEnv1_sampling_dist_2021-10-06-12-16-57', 'gridEnv1_sampling_dist_2021-10-06-12-34-35', 'gridEnv1_sampling_dist_2021-10-06-12-50-59', 'gridEnv1_sampling_dist_2021-10-06-13-03-01', 'gridEnv1_sampling_dist_2021-10-06-13-15-15', 'gridEnv1_sampling_dist_2021-10-06-13-27-00', 'gridEnv1_sampling_dist_2021-10-06-13-50-30', 'gridEnv1_sampling_dist_2021-10-06-14-06-56', 'gridEnv1_sampling_dist_2021-10-06-14-23-43', 'gridEnv1_sampling_dist_2021-10-06-14-41-05', 'gridEnv1_sampling_dist_2021-10-06-14-53-47', 'gridEnv1_sampling_dist_2021-10-06-15-08-37', 'gridEnv1_sampling_dist_2021-10-06-15-28-19', 'gridEnv1_sampling_dist_2021-10-06-15-45-47', 'gridEnv1_sampling_dist_2021-10-06-16-02-02', 'gridEnv1_sampling_dist_2021-10-06-16-23-28', 'gridEnv1_sampling_dist_2021-10-06-16-46-42', 'gridEnv1_sampling_dist_2021-10-06-17-05-58', 'gridEnv1_sampling_dist_2021-10-06-17-22-44', 'gridEnv1_sampling_dist_2021-10-06-17-41-29',
        ],
    },

    'gridEnv4': {
        'OPTIMAL_SAMPLING_DIST_IDS': [
        'gridEnv4_sampling_dist_2021-10-05-10-52-48', 'gridEnv4_sampling_dist_2021-10-05-10-52-50', 'gridEnv4_sampling_dist_2021-10-05-10-52-53', 'gridEnv4_sampling_dist_2021-10-05-10-52-56', 'gridEnv4_sampling_dist_2021-10-05-10-52-59', 'gridEnv4_sampling_dist_2021-10-05-10-53-01', 'gridEnv4_sampling_dist_2021-10-05-10-53-04', 'gridEnv4_sampling_dist_2021-10-05-10-53-07', 'gridEnv4_sampling_dist_2021-10-05-10-53-09', 'gridEnv4_sampling_dist_2021-10-05-10-53-12', 'gridEnv4_sampling_dist_2021-10-05-10-53-15', 'gridEnv4_sampling_dist_2021-10-05-10-53-18', 'gridEnv4_sampling_dist_2021-10-05-10-53-20', 'gridEnv4_sampling_dist_2021-10-05-10-53-23', 'gridEnv4_sampling_dist_2021-10-05-10-53-26', 'gridEnv4_sampling_dist_2021-10-05-10-53-28', 'gridEnv4_sampling_dist_2021-10-05-10-53-31', 'gridEnv4_sampling_dist_2021-10-05-10-53-34', 'gridEnv4_sampling_dist_2021-10-05-10-53-37', 'gridEnv4_sampling_dist_2021-10-05-10-53-39'
        ],
        'OFFLINE_DQN_EXP_IDS': [
        'gridEnv4_offline_dqn_2021-10-05-00-21-27', 'gridEnv4_offline_dqn_2021-10-05-00-32-36', 'gridEnv4_offline_dqn_2021-10-05-00-43-54', 'gridEnv4_offline_dqn_2021-10-05-00-55-09', 'gridEnv4_offline_dqn_2021-10-05-01-06-24', 'gridEnv4_offline_dqn_2021-10-05-01-17-38', 'gridEnv4_offline_dqn_2021-10-05-01-28-51', 'gridEnv4_offline_dqn_2021-10-05-01-40-10', 'gridEnv4_offline_dqn_2021-10-05-01-51-22', 'gridEnv4_offline_dqn_2021-10-05-02-02-37', 'gridEnv4_offline_dqn_2021-10-05-02-13-49', 'gridEnv4_offline_dqn_2021-10-05-02-25-03', 'gridEnv4_offline_dqn_2021-10-05-02-36-15', 'gridEnv4_offline_dqn_2021-10-05-02-47-23', 'gridEnv4_offline_dqn_2021-10-05-02-58-36', 'gridEnv4_offline_dqn_2021-10-05-03-09-47', 'gridEnv4_offline_dqn_2021-10-05-03-21-04', 'gridEnv4_offline_dqn_2021-10-05-03-32-20', 'gridEnv4_offline_dqn_2021-10-05-03-43-33', 'gridEnv4_offline_dqn_2021-10-05-03-54-45', 'gridEnv4_offline_dqn_2021-10-05-04-05-59', 'gridEnv4_offline_dqn_2021-10-05-04-17-11', 'gridEnv4_offline_dqn_2021-10-05-04-28-24', 'gridEnv4_offline_dqn_2021-10-05-04-39-32', 'gridEnv4_offline_dqn_2021-10-05-04-50-41', 'gridEnv4_offline_dqn_2021-10-05-05-01-55', 'gridEnv4_offline_dqn_2021-10-05-05-13-06', 'gridEnv4_offline_dqn_2021-10-05-05-24-17', 'gridEnv4_offline_dqn_2021-10-05-05-35-34', 'gridEnv4_offline_dqn_2021-10-05-05-46-49', 'gridEnv4_offline_dqn_2021-10-05-05-58-04', 'gridEnv4_offline_dqn_2021-10-05-06-09-19', 'gridEnv4_offline_dqn_2021-10-05-06-20-31', 'gridEnv4_offline_dqn_2021-10-05-06-31-41', 'gridEnv4_offline_dqn_2021-10-05-06-42-57', 'gridEnv4_offline_dqn_2021-10-05-06-54-11', 'gridEnv4_offline_dqn_2021-10-05-07-05-21', 'gridEnv4_offline_dqn_2021-10-05-07-16-28', 'gridEnv4_offline_dqn_2021-10-05-07-27-46', 'gridEnv4_offline_dqn_2021-10-05-07-39-01', 'gridEnv4_offline_dqn_2021-10-05-07-50-12', 'gridEnv4_offline_dqn_2021-10-05-08-01-24', 'gridEnv4_offline_dqn_2021-10-05-08-12-36', 'gridEnv4_offline_dqn_2021-10-05-08-23-54', 'gridEnv4_offline_dqn_2021-10-05-08-35-10', 'gridEnv4_offline_dqn_2021-10-05-08-46-24', 'gridEnv4_offline_dqn_2021-10-05-08-57-34', 'gridEnv4_offline_dqn_2021-10-05-09-08-51', 'gridEnv4_offline_dqn_2021-10-05-09-20-01', 'gridEnv4_offline_dqn_2021-10-05-09-32-15',
        'gridEnv4_offline_dqn_2021-10-05-11-23-03', 'gridEnv4_offline_dqn_2021-10-05-11-34-26', 'gridEnv4_offline_dqn_2021-10-05-11-45-50', 'gridEnv4_offline_dqn_2021-10-05-11-57-11', 'gridEnv4_offline_dqn_2021-10-05-12-08-29', 'gridEnv4_offline_dqn_2021-10-05-12-19-55', 'gridEnv4_offline_dqn_2021-10-05-12-31-17', 'gridEnv4_offline_dqn_2021-10-05-12-42-41', 'gridEnv4_offline_dqn_2021-10-05-12-54-02', 'gridEnv4_offline_dqn_2021-10-05-13-05-29', 'gridEnv4_offline_dqn_2021-10-05-13-16-50', 'gridEnv4_offline_dqn_2021-10-05-13-29-45', 'gridEnv4_offline_dqn_2021-10-05-13-43-41', 'gridEnv4_offline_dqn_2021-10-05-13-56-59', 'gridEnv4_offline_dqn_2021-10-05-14-10-58', 'gridEnv4_offline_dqn_2021-10-05-14-24-53', 'gridEnv4_offline_dqn_2021-10-05-14-38-40', 'gridEnv4_offline_dqn_2021-10-05-14-52-35', 'gridEnv4_offline_dqn_2021-10-05-15-06-23', 'gridEnv4_offline_dqn_2021-10-05-15-18-22', 'gridEnv4_offline_dqn_2021-10-05-15-29-44',
        'gridEnv4_offline_dqn_2021-10-05-16-47-38', 'gridEnv4_offline_dqn_2021-10-05-17-05-37', 'gridEnv4_offline_dqn_2021-10-05-17-18-08', 'gridEnv4_offline_dqn_2021-10-05-17-31-01', 'gridEnv4_offline_dqn_2021-10-05-17-43-45', 'gridEnv4_offline_dqn_2021-10-05-17-57-34', 'gridEnv4_offline_dqn_2021-10-05-18-11-52', 'gridEnv4_offline_dqn_2021-10-05-18-25-25', 'gridEnv4_offline_dqn_2021-10-05-18-40-54', 'gridEnv4_offline_dqn_2021-10-05-18-57-27', 'gridEnv4_offline_dqn_2021-10-05-19-11-27', 'gridEnv4_offline_dqn_2021-10-05-19-25-57', 'gridEnv4_offline_dqn_2021-10-05-19-39-04', 'gridEnv4_offline_dqn_2021-10-05-19-52-39', 'gridEnv4_offline_dqn_2021-10-05-20-05-40', 'gridEnv4_offline_dqn_2021-10-05-20-19-22', 'gridEnv4_offline_dqn_2021-10-05-20-32-22', 'gridEnv4_offline_dqn_2021-10-05-20-48-31', 'gridEnv4_offline_dqn_2021-10-05-21-01-20', 'gridEnv4_offline_dqn_2021-10-05-21-14-58', 'gridEnv4_offline_dqn_2021-10-05-21-28-03', 'gridEnv4_offline_dqn_2021-10-05-21-42-52', 'gridEnv4_offline_dqn_2021-10-05-21-57-19', 'gridEnv4_offline_dqn_2021-10-05-22-13-37', 'gridEnv4_offline_dqn_2021-10-05-22-28-02', 'gridEnv4_offline_dqn_2021-10-05-22-41-23', 'gridEnv4_offline_dqn_2021-10-05-22-56-18', 'gridEnv4_offline_dqn_2021-10-05-23-11-37', 'gridEnv4_offline_dqn_2021-10-05-23-33-52', 'gridEnv4_offline_dqn_2021-10-05-23-47-26', 'gridEnv4_offline_dqn_2021-10-06-00-01-58', 'gridEnv4_offline_dqn_2021-10-06-00-14-14',
        ],
        'SAMPLING_DISTS_IDS': [
        'gridEnv4_sampling_dist_2021-10-05-00-21-18', 'gridEnv4_sampling_dist_2021-10-05-00-32-27', 'gridEnv4_sampling_dist_2021-10-05-00-43-46', 'gridEnv4_sampling_dist_2021-10-05-00-55-01', 'gridEnv4_sampling_dist_2021-10-05-01-06-15', 'gridEnv4_sampling_dist_2021-10-05-01-17-29', 'gridEnv4_sampling_dist_2021-10-05-01-28-42', 'gridEnv4_sampling_dist_2021-10-05-01-40-01', 'gridEnv4_sampling_dist_2021-10-05-01-51-14', 'gridEnv4_sampling_dist_2021-10-05-02-02-29', 'gridEnv4_sampling_dist_2021-10-05-02-13-40', 'gridEnv4_sampling_dist_2021-10-05-02-24-54', 'gridEnv4_sampling_dist_2021-10-05-02-36-06', 'gridEnv4_sampling_dist_2021-10-05-02-47-14', 'gridEnv4_sampling_dist_2021-10-05-02-58-27', 'gridEnv4_sampling_dist_2021-10-05-03-09-38', 'gridEnv4_sampling_dist_2021-10-05-03-20-55', 'gridEnv4_sampling_dist_2021-10-05-03-32-11', 'gridEnv4_sampling_dist_2021-10-05-03-43-24', 'gridEnv4_sampling_dist_2021-10-05-03-54-36', 'gridEnv4_sampling_dist_2021-10-05-04-05-51', 'gridEnv4_sampling_dist_2021-10-05-04-17-02', 'gridEnv4_sampling_dist_2021-10-05-04-28-16', 'gridEnv4_sampling_dist_2021-10-05-04-39-24', 'gridEnv4_sampling_dist_2021-10-05-04-50-33', 'gridEnv4_sampling_dist_2021-10-05-05-01-47', 'gridEnv4_sampling_dist_2021-10-05-05-12-57', 'gridEnv4_sampling_dist_2021-10-05-05-24-08', 'gridEnv4_sampling_dist_2021-10-05-05-35-25', 'gridEnv4_sampling_dist_2021-10-05-05-46-40', 'gridEnv4_sampling_dist_2021-10-05-05-57-56', 'gridEnv4_sampling_dist_2021-10-05-06-09-11', 'gridEnv4_sampling_dist_2021-10-05-06-20-22', 'gridEnv4_sampling_dist_2021-10-05-06-31-32', 'gridEnv4_sampling_dist_2021-10-05-06-42-49', 'gridEnv4_sampling_dist_2021-10-05-06-54-02', 'gridEnv4_sampling_dist_2021-10-05-07-05-12', 'gridEnv4_sampling_dist_2021-10-05-07-16-20', 'gridEnv4_sampling_dist_2021-10-05-07-27-37', 'gridEnv4_sampling_dist_2021-10-05-07-38-52', 'gridEnv4_sampling_dist_2021-10-05-07-50-04', 'gridEnv4_sampling_dist_2021-10-05-08-01-16', 'gridEnv4_sampling_dist_2021-10-05-08-12-28', 'gridEnv4_sampling_dist_2021-10-05-08-23-46', 'gridEnv4_sampling_dist_2021-10-05-08-35-02', 'gridEnv4_sampling_dist_2021-10-05-08-46-15', 'gridEnv4_sampling_dist_2021-10-05-08-57-26', 'gridEnv4_sampling_dist_2021-10-05-09-08-42', 'gridEnv4_sampling_dist_2021-10-05-09-19-52', 'gridEnv4_sampling_dist_2021-10-05-09-32-07',
        'gridEnv4_sampling_dist_2021-10-05-11-22-54', 'gridEnv4_sampling_dist_2021-10-05-11-34-17', 'gridEnv4_sampling_dist_2021-10-05-11-45-41', 'gridEnv4_sampling_dist_2021-10-05-11-57-02', 'gridEnv4_sampling_dist_2021-10-05-12-08-21', 'gridEnv4_sampling_dist_2021-10-05-12-19-47', 'gridEnv4_sampling_dist_2021-10-05-12-31-08', 'gridEnv4_sampling_dist_2021-10-05-12-42-32', 'gridEnv4_sampling_dist_2021-10-05-12-53-53', 'gridEnv4_sampling_dist_2021-10-05-13-05-21', 'gridEnv4_sampling_dist_2021-10-05-13-16-41', 'gridEnv4_sampling_dist_2021-10-05-13-29-33', 'gridEnv4_sampling_dist_2021-10-05-13-43-30', 'gridEnv4_sampling_dist_2021-10-05-13-56-47', 'gridEnv4_sampling_dist_2021-10-05-14-10-46', 'gridEnv4_sampling_dist_2021-10-05-14-24-42', 'gridEnv4_sampling_dist_2021-10-05-14-38-28', 'gridEnv4_sampling_dist_2021-10-05-14-52-23', 'gridEnv4_sampling_dist_2021-10-05-15-06-11', 'gridEnv4_sampling_dist_2021-10-05-15-18-13', 'gridEnv4_sampling_dist_2021-10-05-15-29-36',
        'gridEnv4_sampling_dist_2021-10-05-16-47-29', 'gridEnv4_sampling_dist_2021-10-05-17-05-29', 'gridEnv4_sampling_dist_2021-10-05-17-18-00', 'gridEnv4_sampling_dist_2021-10-05-17-30-52', 'gridEnv4_sampling_dist_2021-10-05-17-43-35', 'gridEnv4_sampling_dist_2021-10-05-17-57-25', 'gridEnv4_sampling_dist_2021-10-05-18-11-41', 'gridEnv4_sampling_dist_2021-10-05-18-25-16', 'gridEnv4_sampling_dist_2021-10-05-18-40-45', 'gridEnv4_sampling_dist_2021-10-05-18-57-18', 'gridEnv4_sampling_dist_2021-10-05-19-11-18', 'gridEnv4_sampling_dist_2021-10-05-19-25-47', 'gridEnv4_sampling_dist_2021-10-05-19-38-55', 'gridEnv4_sampling_dist_2021-10-05-19-52-30', 'gridEnv4_sampling_dist_2021-10-05-20-05-31', 'gridEnv4_sampling_dist_2021-10-05-20-19-12', 'gridEnv4_sampling_dist_2021-10-05-20-32-13', 'gridEnv4_sampling_dist_2021-10-05-20-48-22', 'gridEnv4_sampling_dist_2021-10-05-21-01-11', 'gridEnv4_sampling_dist_2021-10-05-21-14-50', 'gridEnv4_sampling_dist_2021-10-05-21-27-55', 'gridEnv4_sampling_dist_2021-10-05-21-42-42', 'gridEnv4_sampling_dist_2021-10-05-21-57-11', 'gridEnv4_sampling_dist_2021-10-05-22-13-27', 'gridEnv4_sampling_dist_2021-10-05-22-27-52', 'gridEnv4_sampling_dist_2021-10-05-22-41-13', 'gridEnv4_sampling_dist_2021-10-05-22-56-08', 'gridEnv4_sampling_dist_2021-10-05-23-11-26', 'gridEnv4_sampling_dist_2021-10-05-23-33-42', 'gridEnv4_sampling_dist_2021-10-05-23-47-16', 'gridEnv4_sampling_dist_2021-10-06-00-01-47', 'gridEnv4_sampling_dist_2021-10-06-00-14-05' 
        ],
    },

    'multiPathsEnv': {
        'OPTIMAL_SAMPLING_DIST_IDS': [
        'multiPathsEnv_sampling_dist_2021-10-07-00-49-56', 'multiPathsEnv_sampling_dist_2021-10-07-00-49-58', 'multiPathsEnv_sampling_dist_2021-10-07-00-49-59', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-01', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-03', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-04', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-06', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-07', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-09', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-11', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-12', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-14', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-15', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-17', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-19', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-20', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-22', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-24', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-25', 'multiPathsEnv_sampling_dist_2021-10-07-00-50-27',
        ],
        'OFFLINE_DQN_EXP_IDS': [
        'multiPathsEnv_offline_dqn_2021-10-06-19-56-37', 'multiPathsEnv_offline_dqn_2021-10-06-20-03-46', 'multiPathsEnv_offline_dqn_2021-10-06-20-10-53', 'multiPathsEnv_offline_dqn_2021-10-06-20-18-08', 'multiPathsEnv_offline_dqn_2021-10-06-20-25-24', 'multiPathsEnv_offline_dqn_2021-10-06-20-33-11', 'multiPathsEnv_offline_dqn_2021-10-06-20-41-57', 'multiPathsEnv_offline_dqn_2021-10-06-20-50-34', 'multiPathsEnv_offline_dqn_2021-10-06-20-59-13', 'multiPathsEnv_offline_dqn_2021-10-06-21-08-00', 'multiPathsEnv_offline_dqn_2021-10-06-21-16-54', 'multiPathsEnv_offline_dqn_2021-10-06-21-25-29', 'multiPathsEnv_offline_dqn_2021-10-06-21-34-07', 'multiPathsEnv_offline_dqn_2021-10-06-21-42-46', 'multiPathsEnv_offline_dqn_2021-10-06-21-51-44', 'multiPathsEnv_offline_dqn_2021-10-06-22-00-21', 'multiPathsEnv_offline_dqn_2021-10-06-22-08-58', 'multiPathsEnv_offline_dqn_2021-10-06-22-17-33', 'multiPathsEnv_offline_dqn_2021-10-06-22-26-37', 'multiPathsEnv_offline_dqn_2021-10-06-22-35-19', 'multiPathsEnv_offline_dqn_2021-10-06-22-43-52', 'multiPathsEnv_offline_dqn_2021-10-06-22-52-31', 'multiPathsEnv_offline_dqn_2021-10-06-23-01-23', 'multiPathsEnv_offline_dqn_2021-10-06-23-10-10', 'multiPathsEnv_offline_dqn_2021-10-06-23-18-44',
        'multiPathsEnv_offline_dqn_2021-10-10-14-31-58', 'multiPathsEnv_offline_dqn_2021-10-10-14-43-35', 'multiPathsEnv_offline_dqn_2021-10-10-14-55-59', 'multiPathsEnv_offline_dqn_2021-10-10-15-07-12', 'multiPathsEnv_offline_dqn_2021-10-10-15-21-24', 'multiPathsEnv_offline_dqn_2021-10-10-15-33-08', 'multiPathsEnv_offline_dqn_2021-10-10-15-47-56', 'multiPathsEnv_offline_dqn_2021-10-10-16-00-01', 'multiPathsEnv_offline_dqn_2021-10-10-16-12-46', 'multiPathsEnv_offline_dqn_2021-10-10-16-26-40', 'multiPathsEnv_offline_dqn_2021-10-10-16-39-24', 'multiPathsEnv_offline_dqn_2021-10-10-16-53-06', 'multiPathsEnv_offline_dqn_2021-10-10-17-11-00', 'multiPathsEnv_offline_dqn_2021-10-10-17-22-59', 'multiPathsEnv_offline_dqn_2021-10-10-17-37-00', 'multiPathsEnv_offline_dqn_2021-10-10-17-48-32', 'multiPathsEnv_offline_dqn_2021-10-10-18-00-54', 'multiPathsEnv_offline_dqn_2021-10-10-18-15-23', 'multiPathsEnv_offline_dqn_2021-10-10-18-27-55', 'multiPathsEnv_offline_dqn_2021-10-10-18-42-42', 'multiPathsEnv_offline_dqn_2021-10-10-18-58-31', 'multiPathsEnv_offline_dqn_2021-10-10-19-10-24', 'multiPathsEnv_offline_dqn_2021-10-10-19-22-33', 'multiPathsEnv_offline_dqn_2021-10-10-19-36-48', 'multiPathsEnv_offline_dqn_2021-10-10-19-48-34',
        'multiPathsEnv_offline_dqn_2021-10-11-18-06-15', 'multiPathsEnv_offline_dqn_2021-10-11-18-59-51', 'multiPathsEnv_offline_dqn_2021-10-11-19-45-27', 'multiPathsEnv_offline_dqn_2021-10-11-20-27-45', 'multiPathsEnv_offline_dqn_2021-10-11-21-10-34', 'multiPathsEnv_offline_dqn_2021-10-11-21-53-43', 'multiPathsEnv_offline_dqn_2021-10-11-22-38-45', 'multiPathsEnv_offline_dqn_2021-10-11-23-23-11', 'multiPathsEnv_offline_dqn_2021-10-12-00-16-06', 'multiPathsEnv_offline_dqn_2021-10-12-01-14-30', 'multiPathsEnv_offline_dqn_2021-10-12-01-57-54', 'multiPathsEnv_offline_dqn_2021-10-12-02-49-02', 'multiPathsEnv_offline_dqn_2021-10-12-03-31-53', 'multiPathsEnv_offline_dqn_2021-10-12-04-17-08', 'multiPathsEnv_offline_dqn_2021-10-12-04-59-58', 'multiPathsEnv_offline_dqn_2021-10-12-05-43-03', 'multiPathsEnv_offline_dqn_2021-10-12-06-34-28', 'multiPathsEnv_offline_dqn_2021-10-12-07-29-35', 'multiPathsEnv_offline_dqn_2021-10-12-08-12-13', 'multiPathsEnv_offline_dqn_2021-10-12-09-03-15', 'multiPathsEnv_offline_dqn_2021-10-12-09-46-17', 'multiPathsEnv_offline_dqn_2021-10-12-10-37-22',
        ],
        'SAMPLING_DISTS_IDS': [
        'multiPathsEnv_sampling_dist_2021-10-06-19-56-33', 'multiPathsEnv_sampling_dist_2021-10-06-20-03-42', 'multiPathsEnv_sampling_dist_2021-10-06-20-10-48', 'multiPathsEnv_sampling_dist_2021-10-06-20-18-04', 'multiPathsEnv_sampling_dist_2021-10-06-20-25-20', 'multiPathsEnv_sampling_dist_2021-10-06-20-33-06', 'multiPathsEnv_sampling_dist_2021-10-06-20-41-52', 'multiPathsEnv_sampling_dist_2021-10-06-20-50-30', 'multiPathsEnv_sampling_dist_2021-10-06-20-59-08', 'multiPathsEnv_sampling_dist_2021-10-06-21-07-54', 'multiPathsEnv_sampling_dist_2021-10-06-21-16-49', 'multiPathsEnv_sampling_dist_2021-10-06-21-25-25', 'multiPathsEnv_sampling_dist_2021-10-06-21-34-03', 'multiPathsEnv_sampling_dist_2021-10-06-21-42-41', 'multiPathsEnv_sampling_dist_2021-10-06-21-51-39', 'multiPathsEnv_sampling_dist_2021-10-06-22-00-16', 'multiPathsEnv_sampling_dist_2021-10-06-22-08-53', 'multiPathsEnv_sampling_dist_2021-10-06-22-17-28', 'multiPathsEnv_sampling_dist_2021-10-06-22-26-31', 'multiPathsEnv_sampling_dist_2021-10-06-22-35-14', 'multiPathsEnv_sampling_dist_2021-10-06-22-43-48', 'multiPathsEnv_sampling_dist_2021-10-06-22-52-26', 'multiPathsEnv_sampling_dist_2021-10-06-23-01-18', 'multiPathsEnv_sampling_dist_2021-10-06-23-10-05', 'multiPathsEnv_sampling_dist_2021-10-06-23-18-39',
        'multiPathsEnv_sampling_dist_2021-10-10-14-31-52', 'multiPathsEnv_sampling_dist_2021-10-10-14-43-30', 'multiPathsEnv_sampling_dist_2021-10-10-14-55-54', 'multiPathsEnv_sampling_dist_2021-10-10-15-07-06', 'multiPathsEnv_sampling_dist_2021-10-10-15-21-19', 'multiPathsEnv_sampling_dist_2021-10-10-15-33-02', 'multiPathsEnv_sampling_dist_2021-10-10-15-47-50', 'multiPathsEnv_sampling_dist_2021-10-10-15-59-56', 'multiPathsEnv_sampling_dist_2021-10-10-16-12-41', 'multiPathsEnv_sampling_dist_2021-10-10-16-26-34', 'multiPathsEnv_sampling_dist_2021-10-10-16-39-18', 'multiPathsEnv_sampling_dist_2021-10-10-16-53-01', 'multiPathsEnv_sampling_dist_2021-10-10-17-10-55', 'multiPathsEnv_sampling_dist_2021-10-10-17-22-54', 'multiPathsEnv_sampling_dist_2021-10-10-17-36-55', 'multiPathsEnv_sampling_dist_2021-10-10-17-48-27', 'multiPathsEnv_sampling_dist_2021-10-10-18-00-48', 'multiPathsEnv_sampling_dist_2021-10-10-18-15-17', 'multiPathsEnv_sampling_dist_2021-10-10-18-27-50', 'multiPathsEnv_sampling_dist_2021-10-10-18-42-37', 'multiPathsEnv_sampling_dist_2021-10-10-18-58-26', 'multiPathsEnv_sampling_dist_2021-10-10-19-10-19', 'multiPathsEnv_sampling_dist_2021-10-10-19-22-27', 'multiPathsEnv_sampling_dist_2021-10-10-19-36-43', 'multiPathsEnv_sampling_dist_2021-10-10-19-48-29',
        'multiPathsEnv_sampling_dist_2021-10-11-18-06-10', 'multiPathsEnv_sampling_dist_2021-10-11-18-59-46', 'multiPathsEnv_sampling_dist_2021-10-11-19-45-22', 'multiPathsEnv_sampling_dist_2021-10-11-20-27-40', 'multiPathsEnv_sampling_dist_2021-10-11-21-10-29', 'multiPathsEnv_sampling_dist_2021-10-11-21-53-38', 'multiPathsEnv_sampling_dist_2021-10-11-22-38-39', 'multiPathsEnv_sampling_dist_2021-10-11-23-23-06', 'multiPathsEnv_sampling_dist_2021-10-12-00-16-00', 'multiPathsEnv_sampling_dist_2021-10-12-01-14-24', 'multiPathsEnv_sampling_dist_2021-10-12-01-57-48', 'multiPathsEnv_sampling_dist_2021-10-12-02-48-57', 'multiPathsEnv_sampling_dist_2021-10-12-03-31-47', 'multiPathsEnv_sampling_dist_2021-10-12-04-17-03', 'multiPathsEnv_sampling_dist_2021-10-12-04-59-52', 'multiPathsEnv_sampling_dist_2021-10-12-05-42-57', 'multiPathsEnv_sampling_dist_2021-10-12-06-34-23', 'multiPathsEnv_sampling_dist_2021-10-12-07-29-30', 'multiPathsEnv_sampling_dist_2021-10-12-08-12-08', 'multiPathsEnv_sampling_dist_2021-10-12-09-03-09', 'multiPathsEnv_sampling_dist_2021-10-12-09-46-11', 'multiPathsEnv_sampling_dist_2021-10-12-10-37-17',
        ],
    },

    'pendulum': {
        'OPTIMAL_SAMPLING_DIST_IDS': [
        'pendulum_sampling_dist_2021-10-07-18-40-46', 'pendulum_sampling_dist_2021-10-07-18-40-51', 'pendulum_sampling_dist_2021-10-07-18-40-57', 'pendulum_sampling_dist_2021-10-07-18-41-03', 'pendulum_sampling_dist_2021-10-07-18-41-09', 'pendulum_sampling_dist_2021-10-07-18-41-14', 'pendulum_sampling_dist_2021-10-07-18-41-20', 'pendulum_sampling_dist_2021-10-07-18-41-26', 'pendulum_sampling_dist_2021-10-07-18-41-32', 'pendulum_sampling_dist_2021-10-07-18-41-37', 'pendulum_sampling_dist_2021-10-07-18-41-43', 'pendulum_sampling_dist_2021-10-07-18-41-49', 'pendulum_sampling_dist_2021-10-07-18-41-55', 'pendulum_sampling_dist_2021-10-07-18-42-00', 'pendulum_sampling_dist_2021-10-07-18-42-06', 'pendulum_sampling_dist_2021-10-07-18-42-12', 'pendulum_sampling_dist_2021-10-07-18-42-18', 'pendulum_sampling_dist_2021-10-07-18-42-23', 'pendulum_sampling_dist_2021-10-07-18-42-29', 'pendulum_sampling_dist_2021-10-07-18-42-35'
        ],
        'OFFLINE_DQN_EXP_IDS': [
        'pendulum_offline_dqn_2021-10-07-09-55-12', 'pendulum_offline_dqn_2021-10-07-10-09-51', 'pendulum_offline_dqn_2021-10-07-10-24-12', 'pendulum_offline_dqn_2021-10-07-10-38-38', 'pendulum_offline_dqn_2021-10-07-10-53-21', 'pendulum_offline_dqn_2021-10-07-11-07-51', 'pendulum_offline_dqn_2021-10-07-11-22-26', 'pendulum_offline_dqn_2021-10-07-11-39-56', 'pendulum_offline_dqn_2021-10-07-11-56-36', 'pendulum_offline_dqn_2021-10-07-12-13-42', 'pendulum_offline_dqn_2021-10-07-12-28-57', 'pendulum_offline_dqn_2021-10-07-12-44-44', 'pendulum_offline_dqn_2021-10-07-13-01-25', 'pendulum_offline_dqn_2021-10-07-13-19-19', 'pendulum_offline_dqn_2021-10-07-13-35-27', 'pendulum_offline_dqn_2021-10-07-13-50-17', 'pendulum_offline_dqn_2021-10-07-14-04-50', 'pendulum_offline_dqn_2021-10-07-14-20-02', 'pendulum_offline_dqn_2021-10-07-14-36-46', 'pendulum_offline_dqn_2021-10-07-14-52-42', 'pendulum_offline_dqn_2021-10-07-15-09-27', 'pendulum_offline_dqn_2021-10-07-15-26-29', 'pendulum_offline_dqn_2021-10-07-15-43-12', 'pendulum_offline_dqn_2021-10-07-16-00-20', 'pendulum_offline_dqn_2021-10-07-16-16-37', 'pendulum_offline_dqn_2021-10-07-16-31-11', 'pendulum_offline_dqn_2021-10-07-16-48-12', 'pendulum_offline_dqn_2021-10-07-17-05-35', 'pendulum_offline_dqn_2021-10-07-17-22-12', 'pendulum_offline_dqn_2021-10-07-17-38-49',
        'pendulum_offline_dqn_2021-10-07-18-57-22', 'pendulum_offline_dqn_2021-10-07-19-15-22', 'pendulum_offline_dqn_2021-10-07-19-32-08', 'pendulum_offline_dqn_2021-10-07-19-48-33', 'pendulum_offline_dqn_2021-10-07-20-05-14', 'pendulum_offline_dqn_2021-10-07-20-21-47', 'pendulum_offline_dqn_2021-10-07-20-36-46', 'pendulum_offline_dqn_2021-10-07-20-51-14', 'pendulum_offline_dqn_2021-10-07-21-05-49', 'pendulum_offline_dqn_2021-10-07-21-21-35', 'pendulum_offline_dqn_2021-10-07-21-39-16', 'pendulum_offline_dqn_2021-10-07-21-56-18', 'pendulum_offline_dqn_2021-10-07-22-13-17', 'pendulum_offline_dqn_2021-10-07-22-31-00', 'pendulum_offline_dqn_2021-10-07-22-45-49', 'pendulum_offline_dqn_2021-10-07-23-00-26', 'pendulum_offline_dqn_2021-10-07-23-15-25', 'pendulum_offline_dqn_2021-10-07-23-31-36', 'pendulum_offline_dqn_2021-10-07-23-46-32', 'pendulum_offline_dqn_2021-10-08-00-01-08', 'pendulum_offline_dqn_2021-10-08-00-15-44', 'pendulum_offline_dqn_2021-10-08-00-30-20', 'pendulum_offline_dqn_2021-10-08-00-46-15', 'pendulum_offline_dqn_2021-10-08-01-01-06', 'pendulum_offline_dqn_2021-10-08-01-16-22', 'pendulum_offline_dqn_2021-10-08-01-33-44', 'pendulum_offline_dqn_2021-10-08-01-49-16', 'pendulum_offline_dqn_2021-10-08-02-03-59', 'pendulum_offline_dqn_2021-10-08-02-18-30', 'pendulum_offline_dqn_2021-10-08-02-32-59',
        'pendulum_offline_dqn_2021-10-08-12-12-44', 'pendulum_offline_dqn_2021-10-08-12-40-29', 'pendulum_offline_dqn_2021-10-08-13-07-06', 'pendulum_offline_dqn_2021-10-08-13-41-19', 'pendulum_offline_dqn_2021-10-08-13-58-05', 'pendulum_offline_dqn_2021-10-08-14-33-25', 'pendulum_offline_dqn_2021-10-08-14-52-25', 'pendulum_offline_dqn_2021-10-08-15-16-57', 'pendulum_offline_dqn_2021-10-08-15-33-13', 'pendulum_offline_dqn_2021-10-08-15-50-59',
        'pendulum_offline_dqn_2021-10-10-11-18-55', 'pendulum_offline_dqn_2021-10-10-11-35-46', 'pendulum_offline_dqn_2021-10-10-11-51-51', 'pendulum_offline_dqn_2021-10-10-12-08-12', 'pendulum_offline_dqn_2021-10-10-12-23-28', 'pendulum_offline_dqn_2021-10-10-12-38-33', 'pendulum_offline_dqn_2021-10-10-12-53-30', 'pendulum_offline_dqn_2021-10-10-13-08-36', 'pendulum_offline_dqn_2021-10-10-13-23-23', 'pendulum_offline_dqn_2021-10-10-13-38-36',
        ],
        'SAMPLING_DISTS_IDS': [
        'pendulum_sampling_dist_2021-10-07-09-54-52', 'pendulum_sampling_dist_2021-10-07-10-09-30', 'pendulum_sampling_dist_2021-10-07-10-23-51', 'pendulum_sampling_dist_2021-10-07-10-38-17', 'pendulum_sampling_dist_2021-10-07-10-53-00', 'pendulum_sampling_dist_2021-10-07-11-07-30', 'pendulum_sampling_dist_2021-10-07-11-22-04', 'pendulum_sampling_dist_2021-10-07-11-39-34', 'pendulum_sampling_dist_2021-10-07-11-56-14', 'pendulum_sampling_dist_2021-10-07-12-13-21', 'pendulum_sampling_dist_2021-10-07-12-28-36', 'pendulum_sampling_dist_2021-10-07-12-44-22', 'pendulum_sampling_dist_2021-10-07-13-01-03', 'pendulum_sampling_dist_2021-10-07-13-18-57', 'pendulum_sampling_dist_2021-10-07-13-35-06', 'pendulum_sampling_dist_2021-10-07-13-49-56', 'pendulum_sampling_dist_2021-10-07-14-04-29', 'pendulum_sampling_dist_2021-10-07-14-19-40', 'pendulum_sampling_dist_2021-10-07-14-36-25', 'pendulum_sampling_dist_2021-10-07-14-52-20', 'pendulum_sampling_dist_2021-10-07-15-09-04', 'pendulum_sampling_dist_2021-10-07-15-26-07', 'pendulum_sampling_dist_2021-10-07-15-42-50', 'pendulum_sampling_dist_2021-10-07-15-59-58', 'pendulum_sampling_dist_2021-10-07-16-16-16', 'pendulum_sampling_dist_2021-10-07-16-30-50', 'pendulum_sampling_dist_2021-10-07-16-47-50', 'pendulum_sampling_dist_2021-10-07-17-05-12', 'pendulum_sampling_dist_2021-10-07-17-21-51', 'pendulum_sampling_dist_2021-10-07-17-38-27',
        'pendulum_sampling_dist_2021-10-07-18-57-01', 'pendulum_sampling_dist_2021-10-07-19-15-01', 'pendulum_sampling_dist_2021-10-07-19-31-46', 'pendulum_sampling_dist_2021-10-07-19-48-12', 'pendulum_sampling_dist_2021-10-07-20-04-53', 'pendulum_sampling_dist_2021-10-07-20-21-26', 'pendulum_sampling_dist_2021-10-07-20-36-25', 'pendulum_sampling_dist_2021-10-07-20-50-53', 'pendulum_sampling_dist_2021-10-07-21-05-28', 'pendulum_sampling_dist_2021-10-07-21-21-13', 'pendulum_sampling_dist_2021-10-07-21-38-54', 'pendulum_sampling_dist_2021-10-07-21-55-57', 'pendulum_sampling_dist_2021-10-07-22-12-56', 'pendulum_sampling_dist_2021-10-07-22-30-39', 'pendulum_sampling_dist_2021-10-07-22-45-28', 'pendulum_sampling_dist_2021-10-07-23-00-05', 'pendulum_sampling_dist_2021-10-07-23-15-05', 'pendulum_sampling_dist_2021-10-07-23-31-14', 'pendulum_sampling_dist_2021-10-07-23-46-11', 'pendulum_sampling_dist_2021-10-08-00-00-47', 'pendulum_sampling_dist_2021-10-08-00-15-23', 'pendulum_sampling_dist_2021-10-08-00-29-59', 'pendulum_sampling_dist_2021-10-08-00-45-54', 'pendulum_sampling_dist_2021-10-08-01-00-46', 'pendulum_sampling_dist_2021-10-08-01-16-01', 'pendulum_sampling_dist_2021-10-08-01-33-23', 'pendulum_sampling_dist_2021-10-08-01-48-55', 'pendulum_sampling_dist_2021-10-08-02-03-38', 'pendulum_sampling_dist_2021-10-08-02-18-09', 'pendulum_sampling_dist_2021-10-08-02-32-38',
        'pendulum_sampling_dist_2021-10-08-12-12-23', 'pendulum_sampling_dist_2021-10-08-12-40-08', 'pendulum_sampling_dist_2021-10-08-13-06-45', 'pendulum_sampling_dist_2021-10-08-13-40-57', 'pendulum_sampling_dist_2021-10-08-13-57-44', 'pendulum_sampling_dist_2021-10-08-14-33-03', 'pendulum_sampling_dist_2021-10-08-14-52-04', 'pendulum_sampling_dist_2021-10-08-15-16-36', 'pendulum_sampling_dist_2021-10-08-15-32-52', 'pendulum_sampling_dist_2021-10-08-15-50-38',
        'pendulum_sampling_dist_2021-10-10-11-18-34', 'pendulum_sampling_dist_2021-10-10-11-35-25', 'pendulum_sampling_dist_2021-10-10-11-51-30', 'pendulum_sampling_dist_2021-10-10-12-07-51', 'pendulum_sampling_dist_2021-10-10-12-23-07', 'pendulum_sampling_dist_2021-10-10-12-38-12', 'pendulum_sampling_dist_2021-10-10-12-53-09', 'pendulum_sampling_dist_2021-10-10-13-08-15', 'pendulum_sampling_dist_2021-10-10-13-23-01', 'pendulum_sampling_dist_2021-10-10-13-38-15',
        ],
    },

    #'mountaincar': {
    #},

}

ENVS_LABELS = {
    'gridEnv1': 'Grid 1',
    'gridEnv4': 'Grid 2',
    'multiPathsEnv': 'Multi-path',
    'pendulum': 'Pendulum',
    'mountaincar': 'Mountain car',
}

MAXIMUM_REWARD = {
    'gridEnv1': 36.0,
    'gridEnv4': 41.0,
    'multiPathsEnv': 5.0,
    'pendulum': 42.8, # TODO: check this.
    'mountaincar': 50.0, # TODO: check this.
}

def f_div(x, y):
        y = y + 1e-06
        # return np.dot(y, ((x/y)-1)**2 )
        return np.dot(y, (x/y)**2 - 1)

def smooth(x, y, xgrid):
    samples = np.random.choice(len(x), len(x), replace=True) # resample half of the points.
    y_s = y[samples]
    x_s = x[samples]
    y_sm = lowess(y_s, x_s, frac=0.5, it=5,
                     return_sorted = False)
    # regularly sample it onto the grid
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, 
                                        fill_value='extrapolate')(xgrid)
    return y_grid

#################################################################

FIGURE_X = 6.0
FIGURE_Y = 4.0
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'

def main():
    # Prepare output folder.
    output_folder = PLOTS_FOLDER_PATH + '/distances_plots'
    os.makedirs(output_folder, exist_ok=True)

    data_to_plot = {}
    for env_id, env_data in DATA.items():

        print('env=', env_id)

        optimal_dists = []
        for optimal_dist_id in env_data['OPTIMAL_SAMPLING_DIST_IDS']:
            optimal_dist_path = DATA_FOLDER_PATH + optimal_dist_id + '/data.json'
            print(optimal_dist_path)
            with open(optimal_dist_path, 'r') as f:
                data = json.load(f)
                d = np.array(json.loads(data)['sampling_dist'])
            f.close()
            optimal_dists.append(d)
        print('len(optimal_dists)', len(optimal_dists))

        sampling_dists = []
        for sampling_dist_id in env_data['SAMPLING_DISTS_IDS']:
            sampling_dist_path = DATA_FOLDER_PATH + sampling_dist_id + '/data.json'
            with open(sampling_dist_path, 'r') as f:
                data = json.load(f)
                d = np.array(json.loads(data)['sampling_dist'])
            f.close()
            sampling_dists.append(d)
        print('len(sampling_dists)', len(sampling_dists))

        offline_dqn_metrics = []
        for offline_dqn_exp_id in env_data['OFFLINE_DQN_EXP_IDS']:
            exp_metrics_path = PLOTS_FOLDER_PATH + offline_dqn_exp_id + '/scalar_metrics.json'
            with open(exp_metrics_path, 'r') as f:
                d = json.load(f)
            f.close()
            offline_dqn_metrics.append(d)
        print('len(offline_dqn_metrics)', len(offline_dqn_metrics))

        kl_dists = []
        chi_dists = []
        for sampling_dist in sampling_dists:
            kl_dist = np.min([scipy.stats.entropy(optimal_dist,sampling_dist+1e-06)
                            for optimal_dist in optimal_dists])
            kl_dists.append(kl_dist)
            chi_dist = np.min([f_div(optimal_dist,sampling_dist)
                            for optimal_dist in optimal_dists])
            chi_dists.append(chi_dist)

        data_to_plot[env_id] = {
            'optimal_dists': optimal_dists,
            'sampling_dists': sampling_dists,
            'offline_dqn_metrics': offline_dqn_metrics,
            'kl_dists': kl_dists,
            'chi_dists': chi_dists,
        }

    # Q-values error plot.
    metric = 'qvals_avg_error'

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for env_id, env_plot_data in data_to_plot.items():
        print('env_id=', env_id)

        metric_d = [x[metric] for x in env_plot_data['offline_dqn_metrics']]

        # KL-div. plot.
        Y = [y for _, y in sorted(zip(env_plot_data['kl_dists'], metric_d))]
        X = sorted(env_plot_data['kl_dists'])
        X = np.array(X)
        Y = np.array(Y)

        # xgrid = np.linspace(X.min(), X.max())
        # smooths = np.stack([smooth(X, Y, xgrid) for _ in range(100)]).T
        # p = plt.plot(xgrid, smooths, alpha=0.25)

        # LOWESS smoothing.
        #sm_x, sm_y = lowess(Y, X, frac=0.4, 
        #                        it=3, return_sorted = True).T
        # p = plt.plot(sm_x, sm_y, label=ENVS_LABELS[env_id])

        # Scatter.
        plt.scatter(X, Y, alpha=0.7, label=ENVS_LABELS[env_id]) #, color=p[0].get_color())

    plt.xlabel(r'$\min_{\pi^*}$KL$(d_{\pi^*}||\mu$)')
    plt.ylabel(r'$Q$-values error')
    plt.yscale('log')
    plt.xscale('linear')
    plt.legend(loc=2)
    plt.savefig(f'{output_folder}/{metric}_kl_div.png'.format(), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{metric}_kl_div.pdf'.format(), bbox_inches='tight', pad_inches=0)


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for env_id, env_plot_data in data_to_plot.items():
        print('env_id=', env_id)

        metric_d = [x[metric] for x in env_plot_data['offline_dqn_metrics']]

        # KL-div. plot.
        Y = [y for _, y in sorted(zip(env_plot_data['chi_dists'], metric_d))]
        X = sorted(env_plot_data['chi_dists'])
        X = np.array(X)
        Y = np.array(Y)

        # xgrid = np.linspace(X.min(), X.max())
        # smooths = np.stack([smooth(X, Y, xgrid) for _ in range(100)]).T
        # p = plt.plot(xgrid, smooths, alpha=0.25)

        # LOWESS smoothing.
        #sm_x, sm_y = lowess(Y, X, frac=0.4, 
        #                        it=3, return_sorted = True).T
        # p = plt.plot(sm_x, sm_y, label=ENVS_LABELS[env_id])

        # Scatter.
        plt.scatter(X, Y, alpha=0.7, label=ENVS_LABELS[env_id]) #, color=p[0].get_color())

    plt.xlabel(r'$\min_{\pi^*}\chi^2(d_{\pi^*}||\mu$)')
    plt.ylabel(r'$Q$-values error')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc=2)
    plt.savefig(f'{output_folder}/{metric}_chi_div.png'.format(), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{metric}_chi_div.pdf'.format(), bbox_inches='tight', pad_inches=0)


    # Rewards plot.
    metric = 'rewards_default'

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for env_id, env_plot_data in data_to_plot.items():
        print('env_id=', env_id)

        metric_d = [x[metric] for x in env_plot_data['offline_dqn_metrics']]

        # KL-div. plot.
        Y = [y for _, y in sorted(zip(env_plot_data['kl_dists'], metric_d))]
        X = sorted(env_plot_data['kl_dists'])

        Y = np.array(Y) / MAXIMUM_REWARD[env_id]

        # LOWESS smoothing.
        #sm_x, sm_y = lowess(Y, X, frac=0.5, 
        #                        it=5, return_sorted = True).T
        #p = plt.plot(sm_x, sm_y, label=ENVS_LABELS[env_id], zorder=10)
        #plt.plot(X, Y)
        plt.scatter(X, Y, alpha=0.7, label=ENVS_LABELS[env_id])#, color=p[0].get_color())

    plt.xlabel(r'$\min_{\pi^*}$KL$(d_{\pi^*}||\mu$)')
    plt.ylabel('Normalized reward')
    plt.yscale('linear')
    plt.xscale('linear')
    plt.legend(loc=3)
    plt.savefig(f'{output_folder}/{metric}_kl_div.png'.format(), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{metric}_kl_div.pdf'.format(), bbox_inches='tight', pad_inches=0)


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    for env_id, env_plot_data in data_to_plot.items():
        print('env_id=', env_id)

        metric_d = [x[metric] for x in env_plot_data['offline_dqn_metrics']]

        # KL-div. plot.
        Y = [y for _, y in sorted(zip(env_plot_data['chi_dists'], metric_d))]
        X = sorted(env_plot_data['chi_dists'])

        Y = np.array(Y) / MAXIMUM_REWARD[env_id]

        # LOWESS smoothing.
        #sm_x, sm_y = lowess(Y, X, frac=0.5, 
        #                        it=5, return_sorted = True).T
        #p = plt.plot(sm_x, sm_y, label=ENVS_LABELS[env_id], zorder=10)
        #plt.plot(X, Y)
        plt.scatter(X, Y, alpha=0.7, label=ENVS_LABELS[env_id])#, color=p[0].get_color())

    plt.xlabel(r'$\min_{\pi^*}\chi^2(d_{\pi^*}||\mu$)')
    plt.ylabel('Normalized reward')
    plt.yscale('linear')
    plt.xscale('log')
    plt.legend(loc=3)
    plt.savefig(f'{output_folder}/{metric}_chi_div.png'.format(), bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_folder}/{metric}_chi_div.pdf'.format(), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
