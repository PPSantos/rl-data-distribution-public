import os
import json
import pathlib
import tarfile

import numpy as np
import pandas as pd
from scipy import stats

# Path to folder containing data files.
DATA_FOLDER_PATH_1 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH_1 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/plots/'

# Path to folder containing data files (second folder).
DATA_FOLDER_PATH_2 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/data/ilu_server/'
PLOTS_FOLDER_PATH_2 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/plots/ilu_server/'

# Sampling dists. induced by optimal policies of the MDP.
OPTIMAL_SAMPLING_DISTS = {
    'gridEnv1': 'gridEnv1_sampling_dist_2022-02-14-01-09-51',
    'gridEnv2': 'gridEnv2_sampling_dist_2022-02-14-01-27-08',
    'multiPathEnv': 'multiPathEnv_sampling_dist_2022-02-14-01-28-41',
    'mountaincar': 'mountaincar_sampling_dist_2022-02-14-01-29-14',
    'pendulum': 'pendulum_sampling_dist_2022-02-14-01-30-17',
    'cartpole': 'cartpole_sampling_dist_2022-02-24-12-11-20',
}

EXP_IDS = [
###############
# gridEnv1.
###############
# DQN. (ACME branch data).
# Dirichlet.
'gridEnv1_offline_dqn_2022-02-06-00-50-49', 'gridEnv1_offline_dqn_2022-02-06-00-57-11', 'gridEnv1_offline_dqn_2022-02-06-01-03-31', 'gridEnv1_offline_dqn_2022-02-06-01-09-52', 'gridEnv1_offline_dqn_2022-02-06-01-16-15', 'gridEnv1_offline_dqn_2022-02-06-01-22-34', 'gridEnv1_offline_dqn_2022-02-06-01-28-57', 'gridEnv1_offline_dqn_2022-02-06-01-35-41', 'gridEnv1_offline_dqn_2022-02-06-01-42-03', 'gridEnv1_offline_dqn_2022-02-06-01-48-50', 'gridEnv1_offline_dqn_2022-02-06-01-55-12', 'gridEnv1_offline_dqn_2022-02-06-02-01-35', 'gridEnv1_offline_dqn_2022-02-06-02-08-19', 'gridEnv1_offline_dqn_2022-02-06-02-14-43',
# Eps-greedy.
'gridEnv1_offline_dqn_2022-02-06-02-21-03', 'gridEnv1_offline_dqn_2022-02-06-02-27-22', 'gridEnv1_offline_dqn_2022-02-06-02-33-41', 'gridEnv1_offline_dqn_2022-02-06-02-40-01', 'gridEnv1_offline_dqn_2022-02-06-02-46-20', 'gridEnv1_offline_dqn_2022-02-06-02-52-37', 'gridEnv1_offline_dqn_2022-02-06-02-58-56', 'gridEnv1_offline_dqn_2022-02-06-03-05-14', 'gridEnv1_offline_dqn_2022-02-06-03-11-35', 'gridEnv1_offline_dqn_2022-02-06-03-18-17', 'gridEnv1_offline_dqn_2022-02-06-03-24-38', 'gridEnv1_offline_dqn_2022-02-06-03-30-56', 'gridEnv1_offline_dqn_2022-02-06-03-37-16', 'gridEnv1_offline_dqn_2022-02-06-03-43-34', 'gridEnv1_offline_dqn_2022-02-06-03-49-52', 'gridEnv1_offline_dqn_2022-02-06-03-56-12', 'gridEnv1_offline_dqn_2022-02-06-04-02-54', 'gridEnv1_offline_dqn_2022-02-06-04-09-16', 'gridEnv1_offline_dqn_2022-02-06-04-15-36', 'gridEnv1_offline_dqn_2022-02-06-04-21-53', 'gridEnv1_offline_dqn_2022-02-06-04-28-13', 'gridEnv1_offline_dqn_2022-02-06-04-34-58',
# Boltzmann.
'gridEnv1_offline_dqn_2022-02-06-04-41-18', 'gridEnv1_offline_dqn_2022-02-06-04-47-39', 'gridEnv1_offline_dqn_2022-02-06-04-53-59', 'gridEnv1_offline_dqn_2022-02-06-05-00-20', 'gridEnv1_offline_dqn_2022-02-06-05-07-03', 'gridEnv1_offline_dqn_2022-02-06-05-13-21', 'gridEnv1_offline_dqn_2022-02-06-05-19-42', 'gridEnv1_offline_dqn_2022-02-06-05-26-04', 'gridEnv1_offline_dqn_2022-02-06-05-32-24', 'gridEnv1_offline_dqn_2022-02-06-05-38-43', 'gridEnv1_offline_dqn_2022-02-06-05-45-03', 'gridEnv1_offline_dqn_2022-02-06-05-51-22', 'gridEnv1_offline_dqn_2022-02-06-05-57-43', 'gridEnv1_offline_dqn_2022-02-06-06-04-04', 'gridEnv1_offline_dqn_2022-02-06-06-10-47', 'gridEnv1_offline_dqn_2022-02-06-06-17-35', 'gridEnv1_offline_dqn_2022-02-06-06-24-18', 'gridEnv1_offline_dqn_2022-02-06-06-30-39', 'gridEnv1_offline_dqn_2022-02-06-06-36-57', 'gridEnv1_offline_dqn_2022-02-06-06-43-16', 'gridEnv1_offline_dqn_2022-02-06-06-49-37', 'gridEnv1_offline_dqn_2022-02-06-06-55-58', 'gridEnv1_offline_dqn_2022-02-06-07-02-16', 'gridEnv1_offline_dqn_2022-02-06-07-08-38', 'gridEnv1_offline_dqn_2022-02-06-07-14-58', 'gridEnv1_offline_dqn_2022-02-06-07-21-18', 'gridEnv1_offline_dqn_2022-02-06-07-27-38', 'gridEnv1_offline_dqn_2022-02-06-07-34-00', 'gridEnv1_offline_dqn_2022-02-06-07-40-20', 'gridEnv1_offline_dqn_2022-02-06-07-46-44', 'gridEnv1_offline_dqn_2022-02-06-07-53-06', 'gridEnv1_offline_dqn_2022-02-06-07-59-27', 'gridEnv1_offline_dqn_2022-02-06-08-06-10', 'gridEnv1_offline_dqn_2022-02-06-08-12-53', 'gridEnv1_offline_dqn_2022-02-06-08-19-36', 'gridEnv1_offline_dqn_2022-02-06-08-25-56', 'gridEnv1_offline_dqn_2022-02-06-08-32-41', 'gridEnv1_offline_dqn_2022-02-06-08-39-01', 'gridEnv1_offline_dqn_2022-02-06-08-45-21', 'gridEnv1_offline_dqn_2022-02-06-08-52-04', 'gridEnv1_offline_dqn_2022-02-06-08-58-22', 'gridEnv1_offline_dqn_2022-02-06-09-04-42',

# CQL. (ACME branch data).
# Dirichlet.
'gridEnv1_offline_cql_2022-02-07-00-23-34', 'gridEnv1_offline_cql_2022-02-07-00-30-10', 'gridEnv1_offline_cql_2022-02-07-00-36-45', 'gridEnv1_offline_cql_2022-02-07-00-43-30', 'gridEnv1_offline_cql_2022-02-07-00-50-03', 'gridEnv1_offline_cql_2022-02-07-00-56-40', 'gridEnv1_offline_cql_2022-02-07-01-03-10', 'gridEnv1_offline_cql_2022-02-07-01-09-46', 'gridEnv1_offline_cql_2022-02-07-01-16-23', 'gridEnv1_offline_cql_2022-02-07-01-22-57', 'gridEnv1_offline_cql_2022-02-07-01-29-26', 'gridEnv1_offline_cql_2022-02-07-01-36-01', 'gridEnv1_offline_cql_2022-02-07-01-42-33', 'gridEnv1_offline_cql_2022-02-07-01-49-05',
# Eps-greedy.
'gridEnv1_offline_cql_2022-02-07-03-28-19', 'gridEnv1_offline_cql_2022-02-07-03-35-01', 'gridEnv1_offline_cql_2022-02-07-03-41-34', 'gridEnv1_offline_cql_2022-02-07-03-48-03', 'gridEnv1_offline_cql_2022-02-07-03-54-31', 'gridEnv1_offline_cql_2022-02-07-04-01-00', 'gridEnv1_offline_cql_2022-02-07-04-07-29', 'gridEnv1_offline_cql_2022-02-07-04-14-11', 'gridEnv1_offline_cql_2022-02-07-04-20-53', 'gridEnv1_offline_cql_2022-02-07-04-27-21', 'gridEnv1_offline_cql_2022-02-07-04-33-56', 'gridEnv1_offline_cql_2022-02-07-04-40-26', 'gridEnv1_offline_cql_2022-02-07-04-47-07', 'gridEnv1_offline_cql_2022-02-07-04-53-39', 'gridEnv1_offline_cql_2022-02-07-05-00-06', 'gridEnv1_offline_cql_2022-02-07-05-06-47', 'gridEnv1_offline_cql_2022-02-07-05-13-23', 'gridEnv1_offline_cql_2022-02-07-05-20-07', 'gridEnv1_offline_cql_2022-02-07-05-26-40', 'gridEnv1_offline_cql_2022-02-07-05-33-14', 'gridEnv1_offline_cql_2022-02-07-05-39-56', 'gridEnv1_offline_cql_2022-02-07-05-46-38',
# Boltzmann.
'gridEnv1_offline_cql_2022-02-07-08-17-55', 'gridEnv1_offline_cql_2022-02-07-08-24-26', 'gridEnv1_offline_cql_2022-02-07-08-30-56', 'gridEnv1_offline_cql_2022-02-07-08-37-29', 'gridEnv1_offline_cql_2022-02-07-08-44-01', 'gridEnv1_offline_cql_2022-02-07-08-50-29', 'gridEnv1_offline_cql_2022-02-07-08-57-15', 'gridEnv1_offline_cql_2022-02-07-09-03-41', 'gridEnv1_offline_cql_2022-02-07-09-10-24', 'gridEnv1_offline_cql_2022-02-07-09-16-58', 'gridEnv1_offline_cql_2022-02-07-09-23-34', 'gridEnv1_offline_cql_2022-02-07-09-30-18', 'gridEnv1_offline_cql_2022-02-07-09-37-01', 'gridEnv1_offline_cql_2022-02-07-09-43-37', 'gridEnv1_offline_cql_2022-02-07-09-50-20', 'gridEnv1_offline_cql_2022-02-07-09-56-51', 'gridEnv1_offline_cql_2022-02-07-10-03-34', 'gridEnv1_offline_cql_2022-02-07-10-10-04', 'gridEnv1_offline_cql_2022-02-07-10-16-37', 'gridEnv1_offline_cql_2022-02-07-10-23-23', 'gridEnv1_offline_cql_2022-02-07-10-29-58', 'gridEnv1_offline_cql_2022-02-07-10-36-33', 'gridEnv1_offline_cql_2022-02-07-10-43-16', 'gridEnv1_offline_cql_2022-02-07-10-49-49', 'gridEnv1_offline_cql_2022-02-07-10-56-22', 'gridEnv1_offline_cql_2022-02-07-11-03-05', 'gridEnv1_offline_cql_2022-02-07-11-09-33', 'gridEnv1_offline_cql_2022-02-07-11-16-04', 'gridEnv1_offline_cql_2022-02-07-11-22-47', 'gridEnv1_offline_cql_2022-02-07-11-29-21', 'gridEnv1_offline_cql_2022-02-07-11-35-55', 'gridEnv1_offline_cql_2022-02-07-11-42-24', 'gridEnv1_offline_cql_2022-02-07-11-48-54', 'gridEnv1_offline_cql_2022-02-07-11-55-29', 'gridEnv1_offline_cql_2022-02-07-12-01-58', 'gridEnv1_offline_cql_2022-02-07-12-08-41', 'gridEnv1_offline_cql_2022-02-07-12-15-15', 'gridEnv1_offline_cql_2022-02-07-12-21-28', 'gridEnv1_offline_cql_2022-02-07-12-27-35', 'gridEnv1_offline_cql_2022-02-07-12-33-42', 'gridEnv1_offline_cql_2022-02-07-12-39-51', 'gridEnv1_offline_cql_2022-02-07-12-46-03',

###############
# gridEnv2.
###############
# DQN. (old branch data).
# Dirichlet.
'gridEnv2_offline_dqn_2022-01-22-14-07-39', 'gridEnv2_offline_dqn_2022-01-22-14-14-53', 'gridEnv2_offline_dqn_2022-01-22-14-22-02', 'gridEnv2_offline_dqn_2022-01-22-14-29-02', 'gridEnv2_offline_dqn_2022-01-22-14-36-03', 'gridEnv2_offline_dqn_2022-01-22-14-42-59', 'gridEnv2_offline_dqn_2022-01-22-14-49-54', 'gridEnv2_offline_dqn_2022-01-22-14-56-55', 'gridEnv2_offline_dqn_2022-01-22-15-03-58', 'gridEnv2_offline_dqn_2022-01-22-15-10-54', 'gridEnv2_offline_dqn_2022-01-22-15-17-52', 'gridEnv2_offline_dqn_2022-01-22-15-24-45', 'gridEnv2_offline_dqn_2022-01-22-15-31-45', 'gridEnv2_offline_dqn_2022-01-22-15-38-44',
# Eps-greedy.
'gridEnv2_offline_dqn_2022-01-22-15-45-43', 'gridEnv2_offline_dqn_2022-01-22-15-52-33', 'gridEnv2_offline_dqn_2022-01-22-15-59-38', 'gridEnv2_offline_dqn_2022-01-22-16-06-28', 'gridEnv2_offline_dqn_2022-01-22-16-13-25', 'gridEnv2_offline_dqn_2022-01-22-16-20-14', 'gridEnv2_offline_dqn_2022-01-22-16-27-16', 'gridEnv2_offline_dqn_2022-01-22-16-34-09', 'gridEnv2_offline_dqn_2022-01-22-16-41-12', 'gridEnv2_offline_dqn_2022-01-22-16-48-13', 'gridEnv2_offline_dqn_2022-01-22-16-55-09', 'gridEnv2_offline_dqn_2022-01-22-17-02-03', 'gridEnv2_offline_dqn_2022-01-22-17-08-56', 'gridEnv2_offline_dqn_2022-01-22-17-15-51', 'gridEnv2_offline_dqn_2022-01-22-17-22-52', 'gridEnv2_offline_dqn_2022-01-22-17-29-50', 'gridEnv2_offline_dqn_2022-01-22-17-36-53', 'gridEnv2_offline_dqn_2022-01-22-17-43-47', 'gridEnv2_offline_dqn_2022-01-22-17-50-42', 'gridEnv2_offline_dqn_2022-01-22-17-57-44', 'gridEnv2_offline_dqn_2022-01-22-18-04-37', 'gridEnv2_offline_dqn_2022-01-22-18-13-09',
# Boltzmann.
'gridEnv2_offline_dqn_2022-01-22-18-20-22', 'gridEnv2_offline_dqn_2022-01-22-18-27-28', 'gridEnv2_offline_dqn_2022-01-22-18-34-28', 'gridEnv2_offline_dqn_2022-01-22-18-41-24', 'gridEnv2_offline_dqn_2022-01-22-18-48-29', 'gridEnv2_offline_dqn_2022-01-22-18-55-22', 'gridEnv2_offline_dqn_2022-01-22-19-02-20', 'gridEnv2_offline_dqn_2022-01-22-19-09-19', 'gridEnv2_offline_dqn_2022-01-22-19-16-23', 'gridEnv2_offline_dqn_2022-01-22-19-23-29', 'gridEnv2_offline_dqn_2022-01-22-19-30-34', 'gridEnv2_offline_dqn_2022-01-22-19-37-33', 'gridEnv2_offline_dqn_2022-01-22-19-44-34', 'gridEnv2_offline_dqn_2022-01-22-19-51-30', 'gridEnv2_offline_dqn_2022-01-22-19-58-28', 'gridEnv2_offline_dqn_2022-01-22-20-05-26', 'gridEnv2_offline_dqn_2022-01-22-20-12-28', 'gridEnv2_offline_dqn_2022-01-22-20-19-34', 'gridEnv2_offline_dqn_2022-01-22-20-26-54', 'gridEnv2_offline_dqn_2022-01-22-20-34-06', 'gridEnv2_offline_dqn_2022-01-22-20-41-35', 'gridEnv2_offline_dqn_2022-01-22-20-48-51', 'gridEnv2_offline_dqn_2022-01-22-20-55-58', 'gridEnv2_offline_dqn_2022-01-22-21-02-58', 'gridEnv2_offline_dqn_2022-01-22-21-10-02', 'gridEnv2_offline_dqn_2022-01-22-21-17-11', 'gridEnv2_offline_dqn_2022-01-22-21-24-15', 'gridEnv2_offline_dqn_2022-01-22-21-31-21', 'gridEnv2_offline_dqn_2022-01-22-21-38-22', 'gridEnv2_offline_dqn_2022-01-22-21-45-34', 'gridEnv2_offline_dqn_2022-01-22-21-52-37', 'gridEnv2_offline_dqn_2022-01-22-21-59-43', 'gridEnv2_offline_dqn_2022-01-22-22-06-48', 'gridEnv2_offline_dqn_2022-01-22-22-13-47', 'gridEnv2_offline_dqn_2022-01-22-22-20-53', 'gridEnv2_offline_dqn_2022-01-22-22-27-57', 'gridEnv2_offline_dqn_2022-01-22-22-34-54', 'gridEnv2_offline_dqn_2022-01-22-22-41-52', 'gridEnv2_offline_dqn_2022-01-22-22-48-51', 'gridEnv2_offline_dqn_2022-01-22-22-55-57', 'gridEnv2_offline_dqn_2022-01-22-23-03-03', 'gridEnv2_offline_dqn_2022-01-22-23-09-59',

# CQL. (ACME branch data).
# Dirichlet.
'gridEnv2_offline_cql_2022-02-07-01-55-37', 'gridEnv2_offline_cql_2022-02-07-02-02-12', 'gridEnv2_offline_cql_2022-02-07-02-08-57', 'gridEnv2_offline_cql_2022-02-07-02-15-28', 'gridEnv2_offline_cql_2022-02-07-02-22-13', 'gridEnv2_offline_cql_2022-02-07-02-28-45', 'gridEnv2_offline_cql_2022-02-07-02-35-29', 'gridEnv2_offline_cql_2022-02-07-02-41-59', 'gridEnv2_offline_cql_2022-02-07-02-48-44', 'gridEnv2_offline_cql_2022-02-07-02-55-31', 'gridEnv2_offline_cql_2022-02-07-03-02-05', 'gridEnv2_offline_cql_2022-02-07-03-08-35', 'gridEnv2_offline_cql_2022-02-07-03-15-09', 'gridEnv2_offline_cql_2022-02-07-03-21-46',
# Eps-greedy.
'gridEnv2_offline_cql_2022-02-07-05-53-21', 'gridEnv2_offline_cql_2022-02-07-06-00-02', 'gridEnv2_offline_cql_2022-02-07-06-06-36', 'gridEnv2_offline_cql_2022-02-07-06-13-03', 'gridEnv2_offline_cql_2022-02-07-06-19-30', 'gridEnv2_offline_cql_2022-02-07-06-25-55', 'gridEnv2_offline_cql_2022-02-07-06-32-22', 'gridEnv2_offline_cql_2022-02-07-06-39-04', 'gridEnv2_offline_cql_2022-02-07-06-45-32', 'gridEnv2_offline_cql_2022-02-07-06-51-59', 'gridEnv2_offline_cql_2022-02-07-06-58-26', 'gridEnv2_offline_cql_2022-02-07-07-05-08', 'gridEnv2_offline_cql_2022-02-07-07-11-49', 'gridEnv2_offline_cql_2022-02-07-07-18-32', 'gridEnv2_offline_cql_2022-02-07-07-24-59', 'gridEnv2_offline_cql_2022-02-07-07-31-29', 'gridEnv2_offline_cql_2022-02-07-07-38-01', 'gridEnv2_offline_cql_2022-02-07-07-44-30', 'gridEnv2_offline_cql_2022-02-07-07-51-12', 'gridEnv2_offline_cql_2022-02-07-07-57-46', 'gridEnv2_offline_cql_2022-02-07-08-04-31', 'gridEnv2_offline_cql_2022-02-07-08-11-13',
# Boltzmann.
'gridEnv2_offline_cql_2022-02-07-12-52-46', 'gridEnv2_offline_cql_2022-02-07-12-58-54', 'gridEnv2_offline_cql_2022-02-07-13-05-03', 'gridEnv2_offline_cql_2022-02-07-13-11-09', 'gridEnv2_offline_cql_2022-02-07-13-17-16', 'gridEnv2_offline_cql_2022-02-07-13-23-59', 'gridEnv2_offline_cql_2022-02-07-13-30-06', 'gridEnv2_offline_cql_2022-02-07-13-36-49', 'gridEnv2_offline_cql_2022-02-07-13-42-56', 'gridEnv2_offline_cql_2022-02-07-13-49-39', 'gridEnv2_offline_cql_2022-02-07-13-56-22', 'gridEnv2_offline_cql_2022-02-07-14-02-29', 'gridEnv2_offline_cql_2022-02-07-14-09-12', 'gridEnv2_offline_cql_2022-02-07-14-15-18', 'gridEnv2_offline_cql_2022-02-07-14-21-26', 'gridEnv2_offline_cql_2022-02-07-14-27-34', 'gridEnv2_offline_cql_2022-02-07-14-33-41', 'gridEnv2_offline_cql_2022-02-07-14-40-24', 'gridEnv2_offline_cql_2022-02-07-14-46-31', 'gridEnv2_offline_cql_2022-02-07-14-52-39', 'gridEnv2_offline_cql_2022-02-07-14-58-46', 'gridEnv2_offline_cql_2022-02-07-15-04-56', 'gridEnv2_offline_cql_2022-02-07-15-11-39', 'gridEnv2_offline_cql_2022-02-07-15-18-21', 'gridEnv2_offline_cql_2022-02-07-15-24-28', 'gridEnv2_offline_cql_2022-02-07-15-30-35', 'gridEnv2_offline_cql_2022-02-07-15-36-41', 'gridEnv2_offline_cql_2022-02-07-15-42-48', 'gridEnv2_offline_cql_2022-02-07-15-48-53', 'gridEnv2_offline_cql_2022-02-07-15-55-00', 'gridEnv2_offline_cql_2022-02-07-16-01-07', 'gridEnv2_offline_cql_2022-02-07-16-07-16', 'gridEnv2_offline_cql_2022-02-07-16-13-59', 'gridEnv2_offline_cql_2022-02-07-16-20-42', 'gridEnv2_offline_cql_2022-02-07-16-26-49', 'gridEnv2_offline_cql_2022-02-07-16-32-55', 'gridEnv2_offline_cql_2022-02-07-16-39-07', 'gridEnv2_offline_cql_2022-02-07-16-45-18', 'gridEnv2_offline_cql_2022-02-07-16-51-30', 'gridEnv2_offline_cql_2022-02-07-16-57-41', 'gridEnv2_offline_cql_2022-02-07-17-03-52', 'gridEnv2_offline_cql_2022-02-07-17-10-04',

###############
# multiPathEnv.
###############
# DQN. (old branch data).
# Dirichlet.
'multiPathEnv_offline_dqn_2022-01-24-01-35-30', 'multiPathEnv_offline_dqn_2022-01-24-01-44-13', 'multiPathEnv_offline_dqn_2022-01-24-01-51-04', 'multiPathEnv_offline_dqn_2022-01-24-01-57-49', 'multiPathEnv_offline_dqn_2022-01-24-02-04-37', 'multiPathEnv_offline_dqn_2022-01-24-02-11-24', 'multiPathEnv_offline_dqn_2022-01-24-02-18-17', 'multiPathEnv_offline_dqn_2022-01-24-02-25-16', 'multiPathEnv_offline_dqn_2022-01-24-02-32-08', 'multiPathEnv_offline_dqn_2022-01-24-02-38-56', 'multiPathEnv_offline_dqn_2022-01-24-02-45-47', 'multiPathEnv_offline_dqn_2022-01-24-02-52-37', 'multiPathEnv_offline_dqn_2022-01-24-02-59-25', 'multiPathEnv_offline_dqn_2022-01-24-03-06-12',
# Eps-greedy.
'multiPathEnv_offline_dqn_2022-01-24-03-12-59', 'multiPathEnv_offline_dqn_2022-01-24-03-19-54', 'multiPathEnv_offline_dqn_2022-01-24-03-26-42', 'multiPathEnv_offline_dqn_2022-01-24-03-33-28', 'multiPathEnv_offline_dqn_2022-01-24-03-40-14', 'multiPathEnv_offline_dqn_2022-01-24-03-47-09', 'multiPathEnv_offline_dqn_2022-01-24-03-53-58', 'multiPathEnv_offline_dqn_2022-01-24-04-00-51', 'multiPathEnv_offline_dqn_2022-01-24-04-07-37', 'multiPathEnv_offline_dqn_2022-01-24-04-14-26', 'multiPathEnv_offline_dqn_2022-01-24-04-21-20', 'multiPathEnv_offline_dqn_2022-01-24-04-28-18', 'multiPathEnv_offline_dqn_2022-01-24-04-35-06', 'multiPathEnv_offline_dqn_2022-01-24-04-41-55', 'multiPathEnv_offline_dqn_2022-01-24-04-48-49', 'multiPathEnv_offline_dqn_2022-01-24-04-55-34', 'multiPathEnv_offline_dqn_2022-01-24-05-02-21', 'multiPathEnv_offline_dqn_2022-01-24-05-09-11', 'multiPathEnv_offline_dqn_2022-01-24-05-15-59', 'multiPathEnv_offline_dqn_2022-01-24-05-22-54', 'multiPathEnv_offline_dqn_2022-01-24-05-29-42', 'multiPathEnv_offline_dqn_2022-01-24-05-36-33',
# Boltzmann.
'multiPathEnv_offline_dqn_2022-02-15-20-24-42', 'multiPathEnv_offline_dqn_2022-02-15-20-32-24', 'multiPathEnv_offline_dqn_2022-02-15-20-38-16', 'multiPathEnv_offline_dqn_2022-02-15-20-44-08', 'multiPathEnv_offline_dqn_2022-02-15-20-50-51', 'multiPathEnv_offline_dqn_2022-02-15-20-58-32', 'multiPathEnv_offline_dqn_2022-02-15-21-06-14', 'multiPathEnv_offline_dqn_2022-02-15-21-13-59', 'multiPathEnv_offline_dqn_2022-02-15-21-21-15', 'multiPathEnv_offline_dqn_2022-02-15-21-28-57', 'multiPathEnv_offline_dqn_2022-02-15-21-36-10', 'multiPathEnv_offline_dqn_2022-02-15-21-43-23', 'multiPathEnv_offline_dqn_2022-02-15-21-51-03', 'multiPathEnv_offline_dqn_2022-02-15-21-58-44', 'multiPathEnv_offline_dqn_2022-02-15-22-06-24', 'multiPathEnv_offline_dqn_2022-02-15-22-14-07', 'multiPathEnv_offline_dqn_2022-02-15-22-21-47', 'multiPathEnv_offline_dqn_2022-02-15-22-28-58', 'multiPathEnv_offline_dqn_2022-02-15-22-36-38', 'multiPathEnv_offline_dqn_2022-02-15-22-43-50', 'multiPathEnv_offline_dqn_2022-02-15-22-51-31', 'multiPathEnv_offline_dqn_2022-02-15-22-59-11', 'multiPathEnv_offline_dqn_2022-02-15-23-06-51', 'multiPathEnv_offline_dqn_2022-02-15-23-14-05', 'multiPathEnv_offline_dqn_2022-02-15-23-21-20', 'multiPathEnv_offline_dqn_2022-02-15-23-28-34', 'multiPathEnv_offline_dqn_2022-02-15-23-36-14', 'multiPathEnv_offline_dqn_2022-02-15-23-43-27', 'multiPathEnv_offline_dqn_2022-02-15-23-51-07', 'multiPathEnv_offline_dqn_2022-02-15-23-58-56', 'multiPathEnv_offline_dqn_2022-02-16-00-06-09', 'multiPathEnv_offline_dqn_2022-02-16-00-13-23', 'multiPathEnv_offline_dqn_2022-02-16-00-21-08', 'multiPathEnv_offline_dqn_2022-02-16-00-28-49', 'multiPathEnv_offline_dqn_2022-02-16-00-36-31', 'multiPathEnv_offline_dqn_2022-02-16-00-44-12', 'multiPathEnv_offline_dqn_2022-02-16-00-51-49', 'multiPathEnv_offline_dqn_2022-02-16-00-59-26', 'multiPathEnv_offline_dqn_2022-02-16-01-07-03', 'multiPathEnv_offline_dqn_2022-02-16-01-14-50', 'multiPathEnv_offline_dqn_2022-02-16-01-22-34', 'multiPathEnv_offline_dqn_2022-02-16-01-30-09',

# CQL. (ACME branch data).
# Dirichlet.
'multiPathEnv_offline_cql_2022-02-07-19-47-22', 'multiPathEnv_offline_cql_2022-02-07-19-54-04', 'multiPathEnv_offline_cql_2022-02-07-20-00-06', 'multiPathEnv_offline_cql_2022-02-07-20-06-07', 'multiPathEnv_offline_cql_2022-02-07-20-12-49', 'multiPathEnv_offline_cql_2022-02-07-20-18-52', 'multiPathEnv_offline_cql_2022-02-07-20-24-54', 'multiPathEnv_offline_cql_2022-02-07-20-31-36', 'multiPathEnv_offline_cql_2022-02-07-20-38-17', 'multiPathEnv_offline_cql_2022-02-07-20-44-56', 'multiPathEnv_offline_cql_2022-02-07-20-51-35', 'multiPathEnv_offline_cql_2022-02-07-20-57-39', 'multiPathEnv_offline_cql_2022-02-07-21-03-40', 'multiPathEnv_offline_cql_2022-02-07-21-10-20',
# Eps-greedy.
'multiPathEnv_offline_cql_2022-02-07-21-17-01', 'multiPathEnv_offline_cql_2022-02-07-21-23-41', 'multiPathEnv_offline_cql_2022-02-07-21-30-19', 'multiPathEnv_offline_cql_2022-02-07-21-36-24', 'multiPathEnv_offline_cql_2022-02-07-21-42-25', 'multiPathEnv_offline_cql_2022-02-07-21-48-27', 'multiPathEnv_offline_cql_2022-02-07-21-54-30', 'multiPathEnv_offline_cql_2022-02-07-22-01-10', 'multiPathEnv_offline_cql_2022-02-07-22-07-12', 'multiPathEnv_offline_cql_2022-02-07-22-13-49', 'multiPathEnv_offline_cql_2022-02-07-22-20-27', 'multiPathEnv_offline_cql_2022-02-07-22-26-28', 'multiPathEnv_offline_cql_2022-02-07-22-33-06', 'multiPathEnv_offline_cql_2022-02-07-22-39-46', 'multiPathEnv_offline_cql_2022-02-07-22-45-48', 'multiPathEnv_offline_cql_2022-02-07-22-52-28', 'multiPathEnv_offline_cql_2022-02-07-22-59-06', 'multiPathEnv_offline_cql_2022-02-07-23-05-07', 'multiPathEnv_offline_cql_2022-02-07-23-11-08', 'multiPathEnv_offline_cql_2022-02-07-23-17-11', 'multiPathEnv_offline_cql_2022-02-07-23-23-15', 'multiPathEnv_offline_cql_2022-02-07-23-29-55',
# Boltzmann.
'multiPathEnv_offline_cql_2022-02-07-23-36-32', 'multiPathEnv_offline_cql_2022-02-07-23-43-11', 'multiPathEnv_offline_cql_2022-02-07-23-49-13', 'multiPathEnv_offline_cql_2022-02-07-23-55-16', 'multiPathEnv_offline_cql_2022-02-08-00-01-55', 'multiPathEnv_offline_cql_2022-02-08-00-08-36', 'multiPathEnv_offline_cql_2022-02-08-00-14-38', 'multiPathEnv_offline_cql_2022-02-08-00-20-43', 'multiPathEnv_offline_cql_2022-02-08-00-26-42', 'multiPathEnv_offline_cql_2022-02-08-00-33-21', 'multiPathEnv_offline_cql_2022-02-08-00-40-02', 'multiPathEnv_offline_cql_2022-02-08-00-46-03', 'multiPathEnv_offline_cql_2022-02-08-00-52-44', 'multiPathEnv_offline_cql_2022-02-08-00-58-50', 'multiPathEnv_offline_cql_2022-02-08-01-04-51', 'multiPathEnv_offline_cql_2022-02-08-01-10-53', 'multiPathEnv_offline_cql_2022-02-08-01-16-54', 'multiPathEnv_offline_cql_2022-02-08-01-23-33', 'multiPathEnv_offline_cql_2022-02-08-01-30-12', 'multiPathEnv_offline_cql_2022-02-08-01-36-51', 'multiPathEnv_offline_cql_2022-02-08-01-42-52', 'multiPathEnv_offline_cql_2022-02-08-01-48-52', 'multiPathEnv_offline_cql_2022-02-08-01-55-31', 'multiPathEnv_offline_cql_2022-02-08-02-01-33', 'multiPathEnv_offline_cql_2022-02-08-02-08-12', 'multiPathEnv_offline_cql_2022-02-08-02-14-51', 'multiPathEnv_offline_cql_2022-02-08-02-21-29', 'multiPathEnv_offline_cql_2022-02-08-02-28-08', 'multiPathEnv_offline_cql_2022-02-08-02-34-47', 'multiPathEnv_offline_cql_2022-02-08-02-41-26', 'multiPathEnv_offline_cql_2022-02-08-02-47-26', 'multiPathEnv_offline_cql_2022-02-08-02-54-05', 'multiPathEnv_offline_cql_2022-02-08-03-00-44', 'multiPathEnv_offline_cql_2022-02-08-03-07-25', 'multiPathEnv_offline_cql_2022-02-08-03-13-27', 'multiPathEnv_offline_cql_2022-02-08-03-19-28', 'multiPathEnv_offline_cql_2022-02-08-03-25-28', 'multiPathEnv_offline_cql_2022-02-08-03-32-09', 'multiPathEnv_offline_cql_2022-02-08-03-38-10', 'multiPathEnv_offline_cql_2022-02-08-03-44-49', 'multiPathEnv_offline_cql_2022-02-08-03-50-49', 'multiPathEnv_offline_cql_2022-02-08-03-56-50',

###############
# mountaincar
###############
# DQN. (ACME branch data).
# Dirichlet.
'mountaincar_offline_dqn_2022-02-02-21-38-00', 'mountaincar_offline_dqn_2022-02-02-21-52-33', 'mountaincar_offline_dqn_2022-02-02-22-07-09', 'mountaincar_offline_dqn_2022-02-02-22-21-44', 'mountaincar_offline_dqn_2022-02-02-22-36-15', 'mountaincar_offline_dqn_2022-02-02-22-50-50', 'mountaincar_offline_dqn_2022-02-02-23-05-29', 'mountaincar_offline_dqn_2022-02-02-23-19-59', 'mountaincar_offline_dqn_2022-02-02-23-34-35', 'mountaincar_offline_dqn_2022-02-02-23-49-09', 'mountaincar_offline_dqn_2022-02-03-00-03-43', 'mountaincar_offline_dqn_2022-02-03-00-18-20', 'mountaincar_offline_dqn_2022-02-03-00-32-53', 'mountaincar_offline_dqn_2022-02-03-00-47-28',
# Eps-greedy.
'mountaincar_offline_dqn_2022-02-03-10-50-39', 'mountaincar_offline_dqn_2022-02-03-11-04-59', 'mountaincar_offline_dqn_2022-02-03-11-19-33', 'mountaincar_offline_dqn_2022-02-03-11-33-43', 'mountaincar_offline_dqn_2022-02-03-11-48-04', 'mountaincar_offline_dqn_2022-02-03-12-02-31', 'mountaincar_offline_dqn_2022-02-03-12-16-45', 'mountaincar_offline_dqn_2022-02-03-12-31-02', 'mountaincar_offline_dqn_2022-02-03-12-45-18', 'mountaincar_offline_dqn_2022-02-03-12-59-39', 'mountaincar_offline_dqn_2022-02-03-13-13-58', 'mountaincar_offline_dqn_2022-02-03-13-28-16', 'mountaincar_offline_dqn_2022-02-03-13-42-43', 'mountaincar_offline_dqn_2022-02-03-13-56-55', 'mountaincar_offline_dqn_2022-02-03-14-11-12', 'mountaincar_offline_dqn_2022-02-03-14-25-25', 'mountaincar_offline_dqn_2022-02-03-14-39-38', 'mountaincar_offline_dqn_2022-02-03-14-53-53', 'mountaincar_offline_dqn_2022-02-03-15-08-11', 'mountaincar_offline_dqn_2022-02-03-15-22-26', 'mountaincar_offline_dqn_2022-02-03-15-36-44',
# Boltzmann.
'mountaincar_offline_dqn_2022-02-03-15-51-02', 'mountaincar_offline_dqn_2022-02-03-16-05-18', 'mountaincar_offline_dqn_2022-02-03-16-19-45', 'mountaincar_offline_dqn_2022-02-03-16-34-09', 'mountaincar_offline_dqn_2022-02-03-16-48-34', 'mountaincar_offline_dqn_2022-02-03-17-03-42', 'mountaincar_offline_dqn_2022-02-03-17-18-56', 'mountaincar_offline_dqn_2022-02-03-17-34-08', 'mountaincar_offline_dqn_2022-02-03-17-49-47', 'mountaincar_offline_dqn_2022-02-03-18-04-53', 'mountaincar_offline_dqn_2022-02-03-18-20-00', 'mountaincar_offline_dqn_2022-02-03-18-35-14', 'mountaincar_offline_dqn_2022-02-03-18-50-22', 'mountaincar_offline_dqn_2022-02-03-19-05-26', 'mountaincar_offline_dqn_2022-02-03-19-20-25', 'mountaincar_offline_dqn_2022-02-03-19-35-27', 'mountaincar_offline_dqn_2022-02-03-19-50-31', 'mountaincar_offline_dqn_2022-02-03-20-05-38', 'mountaincar_offline_dqn_2022-02-03-20-20-40', 'mountaincar_offline_dqn_2022-02-03-20-35-43', 'mountaincar_offline_dqn_2022-02-03-20-50-49', 'mountaincar_offline_dqn_2022-02-03-21-05-56', 'mountaincar_offline_dqn_2022-02-03-21-21-00', 'mountaincar_offline_dqn_2022-02-03-21-36-12', 'mountaincar_offline_dqn_2022-02-03-21-51-18', 'mountaincar_offline_dqn_2022-02-03-22-06-47', 'mountaincar_offline_dqn_2022-02-03-22-22-13', 'mountaincar_offline_dqn_2022-02-03-22-37-24', 'mountaincar_offline_dqn_2022-02-03-22-52-32', 'mountaincar_offline_dqn_2022-02-03-23-07-39', 'mountaincar_offline_dqn_2022-02-03-23-22-45', 'mountaincar_offline_dqn_2022-02-03-23-37-57', 'mountaincar_offline_dqn_2022-02-03-23-53-07', 'mountaincar_offline_dqn_2022-02-04-00-08-15', 'mountaincar_offline_dqn_2022-02-04-00-23-23', 'mountaincar_offline_dqn_2022-02-04-00-38-28', 'mountaincar_offline_dqn_2022-02-04-00-53-53', 'mountaincar_offline_dqn_2022-02-04-01-09-18', 'mountaincar_offline_dqn_2022-02-04-01-24-22', 'mountaincar_offline_dqn_2022-02-04-01-39-47', 'mountaincar_offline_dqn_2022-02-04-01-54-55', 'mountaincar_offline_dqn_2022-02-04-02-09-58', 'mountaincar_offline_dqn_2022-02-04-02-24-59',

# CQL. (ACME branch data).
# Dirichlet.
'mountaincar_offline_cql_2022-02-08-11-00-37', 'mountaincar_offline_cql_2022-02-08-11-15-45', 'mountaincar_offline_cql_2022-02-08-11-30-54', 'mountaincar_offline_cql_2022-02-08-11-46-01', 'mountaincar_offline_cql_2022-02-08-12-01-10', 'mountaincar_offline_cql_2022-02-08-12-16-16', 'mountaincar_offline_cql_2022-02-08-12-31-23', 'mountaincar_offline_cql_2022-02-08-12-46-32', 'mountaincar_offline_cql_2022-02-08-13-01-43', 'mountaincar_offline_cql_2022-02-08-13-16-56', 'mountaincar_offline_cql_2022-02-08-13-32-09', 'mountaincar_offline_cql_2022-02-08-13-47-18', 'mountaincar_offline_cql_2022-02-08-14-02-27', 'mountaincar_offline_cql_2022-02-08-14-17-36',
# Eps-greedy.
'mountaincar_offline_cql_2022-02-08-14-32-45', 'mountaincar_offline_cql_2022-02-08-14-47-21', 'mountaincar_offline_cql_2022-02-08-15-02-39', 'mountaincar_offline_cql_2022-02-08-15-17-12', 'mountaincar_offline_cql_2022-02-08-15-31-48', 'mountaincar_offline_cql_2022-02-08-15-46-25', 'mountaincar_offline_cql_2022-02-08-16-01-43', 'mountaincar_offline_cql_2022-02-08-16-16-22', 'mountaincar_offline_cql_2022-02-08-16-31-00', 'mountaincar_offline_cql_2022-02-08-16-45-43', 'mountaincar_offline_cql_2022-02-08-17-00-29', 'mountaincar_offline_cql_2022-02-08-17-15-17', 'mountaincar_offline_cql_2022-02-08-17-29-55', 'mountaincar_offline_cql_2022-02-08-17-44-31', 'mountaincar_offline_cql_2022-02-08-17-59-05', 'mountaincar_offline_cql_2022-02-08-18-13-41', 'mountaincar_offline_cql_2022-02-08-18-28-15', 'mountaincar_offline_cql_2022-02-08-18-42-50', 'mountaincar_offline_cql_2022-02-08-18-57-24', 'mountaincar_offline_cql_2022-02-08-19-12-01', 'mountaincar_offline_cql_2022-02-08-19-26-41', 'mountaincar_offline_cql_2022-02-08-19-41-27',
# Boltzmann.
'mountaincar_offline_cql_2022-02-08-19-56-18', 'mountaincar_offline_cql_2022-02-08-20-11-13', 'mountaincar_offline_cql_2022-02-08-20-26-07', 'mountaincar_offline_cql_2022-02-08-20-41-01', 'mountaincar_offline_cql_2022-02-08-20-55-56', 'mountaincar_offline_cql_2022-02-08-21-10-48', 'mountaincar_offline_cql_2022-02-08-21-25-43', 'mountaincar_offline_cql_2022-02-08-21-40-39', 'mountaincar_offline_cql_2022-02-08-21-55-32', 'mountaincar_offline_cql_2022-02-08-22-10-29', 'mountaincar_offline_cql_2022-02-08-22-25-24', 'mountaincar_offline_cql_2022-02-08-22-40-18', 'mountaincar_offline_cql_2022-02-08-22-55-05', 'mountaincar_offline_cql_2022-02-08-23-09-47', 'mountaincar_offline_cql_2022-02-08-23-24-27', 'mountaincar_offline_cql_2022-02-08-23-39-07', 'mountaincar_offline_cql_2022-02-08-23-53-46', 'mountaincar_offline_cql_2022-02-09-00-08-31', 'mountaincar_offline_cql_2022-02-09-00-23-12', 'mountaincar_offline_cql_2022-02-09-00-37-52', 'mountaincar_offline_cql_2022-02-09-00-52-32', 'mountaincar_offline_cql_2022-02-09-01-07-15', 'mountaincar_offline_cql_2022-02-09-01-22-13', 'mountaincar_offline_cql_2022-02-09-01-37-13', 'mountaincar_offline_cql_2022-02-09-01-52-12', 'mountaincar_offline_cql_2022-02-09-02-07-08', 'mountaincar_offline_cql_2022-02-09-02-22-10', 'mountaincar_offline_cql_2022-02-09-02-37-13', 'mountaincar_offline_cql_2022-02-09-02-52-12', 'mountaincar_offline_cql_2022-02-09-03-07-13', 'mountaincar_offline_cql_2022-02-09-03-22-13', 'mountaincar_offline_cql_2022-02-09-03-37-10', 'mountaincar_offline_cql_2022-02-09-03-52-02', 'mountaincar_offline_cql_2022-02-09-04-06-48', 'mountaincar_offline_cql_2022-02-09-04-21-28', 'mountaincar_offline_cql_2022-02-09-04-36-12', 'mountaincar_offline_cql_2022-02-09-04-51-37', 'mountaincar_offline_cql_2022-02-09-05-06-20', 'mountaincar_offline_cql_2022-02-09-05-21-01', 'mountaincar_offline_cql_2022-02-09-05-35-44', 'mountaincar_offline_cql_2022-02-09-05-50-23', 'mountaincar_offline_cql_2022-02-09-06-05-06',

###############
# pendulum
###############
# DQN. (ACME branch data).
# Dirichlet.
'pendulum_offline_dqn_2022-02-11-00-59-47', 'pendulum_offline_dqn_2022-02-11-01-14-50', 'pendulum_offline_dqn_2022-02-11-01-29-50', 'pendulum_offline_dqn_2022-02-11-01-44-52', 'pendulum_offline_dqn_2022-02-11-01-59-54', 'pendulum_offline_dqn_2022-02-11-02-14-56', 'pendulum_offline_dqn_2022-02-11-02-30-50', 'pendulum_offline_dqn_2022-02-11-02-45-52', 'pendulum_offline_dqn_2022-02-11-03-00-49', 'pendulum_offline_dqn_2022-02-11-03-15-52', 'pendulum_offline_dqn_2022-02-11-03-30-54', 'pendulum_offline_dqn_2022-02-11-03-45-53', 'pendulum_offline_dqn_2022-02-11-04-00-52', 'pendulum_offline_dqn_2022-02-11-04-15-52',
# Eps-greedy.
'pendulum_offline_dqn_2022-02-11-10-36-47', 'pendulum_offline_dqn_2022-02-11-10-51-27', 'pendulum_offline_dqn_2022-02-11-11-06-04', 'pendulum_offline_dqn_2022-02-11-11-20-43', 'pendulum_offline_dqn_2022-02-11-11-35-40', 'pendulum_offline_dqn_2022-02-11-11-50-28', 'pendulum_offline_dqn_2022-02-11-12-05-13', 'pendulum_offline_dqn_2022-02-11-12-19-54', 'pendulum_offline_dqn_2022-02-11-12-34-34', 'pendulum_offline_dqn_2022-02-11-12-49-24', 'pendulum_offline_dqn_2022-02-11-13-06-22', 'pendulum_offline_dqn_2022-02-11-13-24-02', 'pendulum_offline_dqn_2022-02-11-13-41-54', 'pendulum_offline_dqn_2022-02-11-13-59-32', 'pendulum_offline_dqn_2022-02-11-14-17-15', 'pendulum_offline_dqn_2022-02-11-14-34-53', 'pendulum_offline_dqn_2022-02-11-14-52-46', 'pendulum_offline_dqn_2022-02-11-15-10-35', 'pendulum_offline_dqn_2022-02-11-15-28-12', 'pendulum_offline_dqn_2022-02-11-15-45-55', 'pendulum_offline_dqn_2022-02-11-16-03-45', 'pendulum_offline_dqn_2022-02-11-16-19-40',
# Boltzmann.
'pendulum_offline_dqn_2022-02-11-16-34-22', 'pendulum_offline_dqn_2022-02-11-16-49-11', 'pendulum_offline_dqn_2022-02-11-17-05-28', 'pendulum_offline_dqn_2022-02-11-17-20-24', 'pendulum_offline_dqn_2022-02-11-17-35-07', 'pendulum_offline_dqn_2022-02-11-17-49-48', 'pendulum_offline_dqn_2022-02-11-18-04-26', 'pendulum_offline_dqn_2022-02-11-18-19-11', 'pendulum_offline_dqn_2022-02-11-18-33-55', 'pendulum_offline_dqn_2022-02-11-18-48-42', 'pendulum_offline_dqn_2022-02-11-19-03-22', 'pendulum_offline_dqn_2022-02-11-19-17-58', 'pendulum_offline_dqn_2022-02-11-19-32-40', 'pendulum_offline_dqn_2022-02-11-19-47-20', 'pendulum_offline_dqn_2022-02-11-20-02-09', 'pendulum_offline_dqn_2022-02-11-20-16-51', 'pendulum_offline_dqn_2022-02-11-20-31-35', 'pendulum_offline_dqn_2022-02-11-20-46-22', 'pendulum_offline_dqn_2022-02-11-21-01-08', 'pendulum_offline_dqn_2022-02-11-21-15-48', 'pendulum_offline_dqn_2022-02-11-21-30-31', 'pendulum_offline_dqn_2022-02-11-21-45-19', 'pendulum_offline_dqn_2022-02-11-22-00-10', 'pendulum_offline_dqn_2022-02-11-22-14-57', 'pendulum_offline_dqn_2022-02-11-22-29-43', 'pendulum_offline_dqn_2022-02-11-22-44-25', 'pendulum_offline_dqn_2022-02-11-22-59-10', 'pendulum_offline_dqn_2022-02-11-23-13-56', 'pendulum_offline_dqn_2022-02-11-23-28-46', 'pendulum_offline_dqn_2022-02-11-23-43-28', 'pendulum_offline_dqn_2022-02-11-23-58-09', 'pendulum_offline_dqn_2022-02-12-00-12-54', 'pendulum_offline_dqn_2022-02-12-00-27-33', 'pendulum_offline_dqn_2022-02-12-00-42-19', 'pendulum_offline_dqn_2022-02-12-00-57-05', 'pendulum_offline_dqn_2022-02-12-01-11-51', 'pendulum_offline_dqn_2022-02-12-01-26-36', 'pendulum_offline_dqn_2022-02-12-01-41-27', 'pendulum_offline_dqn_2022-02-12-01-56-12', 'pendulum_offline_dqn_2022-02-12-02-11-01', 'pendulum_offline_dqn_2022-02-12-02-25-51', 'pendulum_offline_dqn_2022-02-12-02-40-38',

# CQL. (ACME branch data).
# Dirichlet.
'pendulum_offline_cql_2022-02-12-10-44-02', 'pendulum_offline_cql_2022-02-12-10-59-34', 'pendulum_offline_cql_2022-02-12-11-15-03', 'pendulum_offline_cql_2022-02-12-11-30-29', 'pendulum_offline_cql_2022-02-12-11-46-00', 'pendulum_offline_cql_2022-02-12-12-01-31', 'pendulum_offline_cql_2022-02-12-12-17-02', 'pendulum_offline_cql_2022-02-12-12-32-30', 'pendulum_offline_cql_2022-02-12-12-48-08', 'pendulum_offline_cql_2022-02-12-13-03-43', 'pendulum_offline_cql_2022-02-12-13-19-36', 'pendulum_offline_cql_2022-02-12-13-35-04', 'pendulum_offline_cql_2022-02-12-13-50-35', 'pendulum_offline_cql_2022-02-12-14-06-06',
# Eps-greedy.
'pendulum_offline_cql_2022-02-12-14-21-36', 'pendulum_offline_cql_2022-02-12-14-37-01', 'pendulum_offline_cql_2022-02-12-14-52-01', 'pendulum_offline_cql_2022-02-12-15-07-03', 'pendulum_offline_cql_2022-02-12-15-22-03', 'pendulum_offline_cql_2022-02-12-15-37-00', 'pendulum_offline_cql_2022-02-12-15-52-23', 'pendulum_offline_cql_2022-02-12-16-07-22', 'pendulum_offline_cql_2022-02-12-16-22-24', 'pendulum_offline_cql_2022-02-12-16-37-24', 'pendulum_offline_cql_2022-02-12-16-52-25', 'pendulum_offline_cql_2022-02-12-17-07-26', 'pendulum_offline_cql_2022-02-12-17-22-32', 'pendulum_offline_cql_2022-02-12-17-37-31', 'pendulum_offline_cql_2022-02-12-17-52-34', 'pendulum_offline_cql_2022-02-12-18-07-38', 'pendulum_offline_cql_2022-02-12-18-22-37', 'pendulum_offline_cql_2022-02-12-18-37-38', 'pendulum_offline_cql_2022-02-12-18-52-38', 'pendulum_offline_cql_2022-02-12-19-07-39', 'pendulum_offline_cql_2022-02-12-19-22-43', 'pendulum_offline_cql_2022-02-12-19-37-46',
# Boltzmann.
'pendulum_offline_cql_2022-02-12-19-52-46', 'pendulum_offline_cql_2022-02-12-20-07-54', 'pendulum_offline_cql_2022-02-12-20-23-01', 'pendulum_offline_cql_2022-02-12-20-38-09', 'pendulum_offline_cql_2022-02-12-20-53-15', 'pendulum_offline_cql_2022-02-12-21-08-24', 'pendulum_offline_cql_2022-02-12-21-23-31', 'pendulum_offline_cql_2022-02-12-21-38-40', 'pendulum_offline_cql_2022-02-12-21-53-48', 'pendulum_offline_cql_2022-02-12-22-08-51', 'pendulum_offline_cql_2022-02-12-22-24-00', 'pendulum_offline_cql_2022-02-12-22-39-09', 'pendulum_offline_cql_2022-02-12-22-54-17', 'pendulum_offline_cql_2022-02-12-23-09-24', 'pendulum_offline_cql_2022-02-12-23-24-31', 'pendulum_offline_cql_2022-02-12-23-39-35', 'pendulum_offline_cql_2022-02-12-23-54-40', 'pendulum_offline_cql_2022-02-13-00-09-45', 'pendulum_offline_cql_2022-02-13-00-24-59', 'pendulum_offline_cql_2022-02-13-00-40-05', 'pendulum_offline_cql_2022-02-13-00-55-09', 'pendulum_offline_cql_2022-02-13-01-10-19', 'pendulum_offline_cql_2022-02-13-01-25-31', 'pendulum_offline_cql_2022-02-13-01-40-37', 'pendulum_offline_cql_2022-02-13-01-55-41', 'pendulum_offline_cql_2022-02-13-02-10-52', 'pendulum_offline_cql_2022-02-13-02-26-01', 'pendulum_offline_cql_2022-02-13-02-41-08', 'pendulum_offline_cql_2022-02-13-02-56-18', 'pendulum_offline_cql_2022-02-13-03-11-25', 'pendulum_offline_cql_2022-02-13-03-26-37', 'pendulum_offline_cql_2022-02-13-03-41-45', 'pendulum_offline_cql_2022-02-13-03-56-50', 'pendulum_offline_cql_2022-02-13-04-12-01', 'pendulum_offline_cql_2022-02-13-04-27-10', 'pendulum_offline_cql_2022-02-13-04-42-22', 'pendulum_offline_cql_2022-02-13-04-57-27', 'pendulum_offline_cql_2022-02-13-05-12-30', 'pendulum_offline_cql_2022-02-13-05-28-00', 'pendulum_offline_cql_2022-02-13-05-43-11', 'pendulum_offline_cql_2022-02-13-05-58-19', 'pendulum_offline_cql_2022-02-13-06-13-49',

###############
# cartpole
###############
# DQN. (ACME branch data).
# Dirichlet.
# TODO.
# Eps-greedy.
'cartpole_offline_dqn_2022-02-16-16-01-33', 'cartpole_offline_dqn_2022-02-16-17-58-01', 'cartpole_offline_dqn_2022-02-16-19-45-28', 'cartpole_offline_dqn_2022-02-16-21-34-11', 'cartpole_offline_dqn_2022-02-16-23-23-08', 'cartpole_offline_dqn_2022-02-17-01-12-50', 'cartpole_offline_dqn_2022-02-17-03-02-21', 'cartpole_offline_dqn_2022-02-17-04-51-51', 'cartpole_offline_dqn_2022-02-17-06-41-31', 'cartpole_offline_dqn_2022-02-17-08-30-47', 'cartpole_offline_dqn_2022-02-17-10-20-13', 'cartpole_offline_dqn_2022-02-17-12-09-40', 'cartpole_offline_dqn_2022-02-17-13-58-47', 'cartpole_offline_dqn_2022-02-17-15-49-09', 'cartpole_offline_dqn_2022-02-17-17-39-24', 'cartpole_offline_dqn_2022-02-17-19-29-07', 'cartpole_offline_dqn_2022-02-17-21-19-39', 'cartpole_offline_dqn_2022-02-17-23-11-09', 'cartpole_offline_dqn_2022-02-18-01-02-05', 'cartpole_offline_dqn_2022-02-18-02-52-03', 'cartpole_offline_dqn_2022-02-18-04-42-03', 'cartpole_offline_dqn_2022-02-18-06-32-16',
# Boltzmann.
'cartpole_offline_dqn_2022-02-19-20-33-04', 'cartpole_offline_dqn_2022-02-19-21-44-59', 'cartpole_offline_dqn_2022-02-19-22-56-50', 'cartpole_offline_dqn_2022-02-20-00-08-27', 'cartpole_offline_dqn_2022-02-20-01-19-56', 'cartpole_offline_dqn_2022-02-20-02-32-01', 'cartpole_offline_dqn_2022-02-20-03-43-37', 'cartpole_offline_dqn_2022-02-20-04-55-41', 'cartpole_offline_dqn_2022-02-20-06-07-20', 'cartpole_offline_dqn_2022-02-20-07-19-16', 'cartpole_offline_dqn_2022-02-20-08-31-04', 'cartpole_offline_dqn_2022-02-20-09-42-50', 'cartpole_offline_dqn_2022-02-20-10-54-10', 'cartpole_offline_dqn_2022-02-20-12-05-15', 'cartpole_offline_dqn_2022-02-20-13-16-05', 'cartpole_offline_dqn_2022-02-20-14-27-03', 'cartpole_offline_dqn_2022-02-20-15-38-00', 'cartpole_offline_dqn_2022-02-20-16-49-10', 'cartpole_offline_dqn_2022-02-20-18-00-17', 'cartpole_offline_dqn_2022-02-20-19-11-28', 'cartpole_offline_dqn_2022-02-20-20-22-18', 'cartpole_offline_dqn_2022-02-20-21-33-12', 'cartpole_offline_dqn_2022-02-20-22-45-50', 'cartpole_offline_dqn_2022-02-20-23-59-00', 'cartpole_offline_dqn_2022-02-21-01-11-28', 'cartpole_offline_dqn_2022-02-21-02-23-39', 'cartpole_offline_dqn_2022-02-21-03-35-58', 'cartpole_offline_dqn_2022-02-21-04-48-23', 'cartpole_offline_dqn_2022-02-21-06-01-15', 'cartpole_offline_dqn_2022-02-21-07-13-27', 'cartpole_offline_dqn_2022-02-21-08-25-45', 'cartpole_offline_dqn_2022-02-21-09-38-15', 'cartpole_offline_dqn_2022-02-21-10-50-31', 'cartpole_offline_dqn_2022-02-21-12-03-19', 'cartpole_offline_dqn_2022-02-21-13-16-05', 'cartpole_offline_dqn_2022-02-21-14-28-57', 'cartpole_offline_dqn_2022-02-21-15-41-37', 'cartpole_offline_dqn_2022-02-21-16-54-09', 'cartpole_offline_dqn_2022-02-21-18-06-48', 'cartpole_offline_dqn_2022-02-21-19-19-53', 'cartpole_offline_dqn_2022-02-21-20-32-26', 'cartpole_offline_dqn_2022-02-21-21-45-13',

# CQL. (ACME branch data).
# Dirichlet.
# TODO.
# Eps-greedy.
'cartpole_offline_cql_2022-02-18-11-10-36', 'cartpole_offline_cql_2022-02-18-12-23-47', 'cartpole_offline_cql_2022-02-18-13-45-08', 'cartpole_offline_cql_2022-02-18-14-58-32', 'cartpole_offline_cql_2022-02-18-16-11-01', 'cartpole_offline_cql_2022-02-18-17-23-45', 'cartpole_offline_cql_2022-02-18-18-36-43', 'cartpole_offline_cql_2022-02-18-19-49-50', 'cartpole_offline_cql_2022-02-18-21-02-29', 'cartpole_offline_cql_2022-02-18-22-15-09', 'cartpole_offline_cql_2022-02-18-23-27-58', 'cartpole_offline_cql_2022-02-19-00-40-30', 'cartpole_offline_cql_2022-02-19-01-53-29', 'cartpole_offline_cql_2022-02-19-03-06-23', 'cartpole_offline_cql_2022-02-19-04-19-14', 'cartpole_offline_cql_2022-02-19-05-32-29', 'cartpole_offline_cql_2022-02-19-06-45-40', 'cartpole_offline_cql_2022-02-19-07-58-48', 'cartpole_offline_cql_2022-02-19-09-11-53', 'cartpole_offline_cql_2022-02-19-10-24-51', 'cartpole_offline_cql_2022-02-19-11-37-50', 'cartpole_offline_cql_2022-02-19-12-50-34',
# Boltzmann.
'cartpole_offline_cql_2022-02-22-00-40-20', 'cartpole_offline_cql_2022-02-22-01-53-23', 'cartpole_offline_cql_2022-02-22-03-06-59', 'cartpole_offline_cql_2022-02-22-04-19-47', 'cartpole_offline_cql_2022-02-22-05-33-01', 'cartpole_offline_cql_2022-02-22-06-45-51', 'cartpole_offline_cql_2022-02-22-07-58-49', 'cartpole_offline_cql_2022-02-22-09-12-02', 'cartpole_offline_cql_2022-02-22-10-25-22', 'cartpole_offline_cql_2022-02-22-11-38-45', 'cartpole_offline_cql_2022-02-22-12-51-55', 'cartpole_offline_cql_2022-02-22-14-05-01', 'cartpole_offline_cql_2022-02-22-15-18-09', 'cartpole_offline_cql_2022-02-22-16-31-02', 'cartpole_offline_cql_2022-02-22-17-44-09', 'cartpole_offline_cql_2022-02-22-18-57-40', 'cartpole_offline_cql_2022-02-22-20-11-07', 'cartpole_offline_cql_2022-02-22-21-24-05', 'cartpole_offline_cql_2022-02-22-22-37-30', 'cartpole_offline_cql_2022-02-22-23-50-40', 'cartpole_offline_cql_2022-02-23-01-03-59', 'cartpole_offline_cql_2022-02-23-02-17-05', 'cartpole_offline_cql_2022-02-23-03-30-14', 'cartpole_offline_cql_2022-02-23-04-43-34', 'cartpole_offline_cql_2022-02-23-05-56-48', 'cartpole_offline_cql_2022-02-23-07-10-11', 'cartpole_offline_cql_2022-02-23-08-23-41', 'cartpole_offline_cql_2022-02-23-09-37-18', 'cartpole_offline_cql_2022-02-23-10-51-14', 'cartpole_offline_cql_2022-02-23-12-05-09', 'cartpole_offline_cql_2022-02-23-13-18-58', 'cartpole_offline_cql_2022-02-23-14-32-36', 'cartpole_offline_cql_2022-02-23-15-46-15', 'cartpole_offline_cql_2022-02-23-16-59-49', 'cartpole_offline_cql_2022-02-23-18-13-43', 'cartpole_offline_cql_2022-02-23-19-27-15', 'cartpole_offline_cql_2022-02-23-20-39-39', 'cartpole_offline_cql_2022-02-23-21-54-28', 'cartpole_offline_cql_2022-02-23-23-07-50', 'cartpole_offline_cql_2022-02-24-00-21-17', 'cartpole_offline_cql_2022-02-24-01-34-43', 'cartpole_offline_cql_2022-02-24-02-48-17',
]


def chi_div(x, y):
    y = y + 1e-04
    return np.dot(y, (x/y)**2 - 1)

def data_to_csv(exp_ids):

    print('Parsing data to csv file.')

    # Open sampling dists from optimal policies.
    opt_sampling_dists = {}
    for env_id, sampling_dist_id in OPTIMAL_SAMPLING_DISTS.items():
        optimal_dist_path = DATA_FOLDER_PATH_1 + sampling_dist_id + '/data.json'
        with open(optimal_dist_path, 'r') as f:
            data = json.load(f)
            d = np.array(json.loads(data)['sampling_dists'])
        f.close()
        opt_sampling_dists[env_id] = d

    df_rows = []

    for exp_id in exp_ids:

        row_data = {}

        # Check if file exists and setup paths.
        if os.path.isfile(DATA_FOLDER_PATH_1 + exp_id + '.tar.gz') and \
            os.path.isfile(PLOTS_FOLDER_PATH_1 + exp_id + '/scalar_metrics.json'):
            data_folder_path = DATA_FOLDER_PATH_1
            plots_folder_path = PLOTS_FOLDER_PATH_1
        elif os.path.isfile(DATA_FOLDER_PATH_2 + exp_id + '.tar.gz') and \
            os.path.isfile(PLOTS_FOLDER_PATH_2 + exp_id + '/scalar_metrics.json'):
            data_folder_path = DATA_FOLDER_PATH_2
            plots_folder_path = PLOTS_FOLDER_PATH_2
        else:
            raise FileNotFoundError(f"Unable to find experiment {exp_id} data.")

        # Load experiment arguments to get dataset parameters.
        exp_args_path = plots_folder_path + exp_id + '/args.json'
        with open(exp_args_path, 'r') as f:
            args = json.load(f)
        f.close()

        row_data['id'] = exp_id
        row_data['env_id'] = args['env_name']
        row_data['algo_id'] = args['train_args']['algo']
        row_data['dataset_type_id'] = args['dataset_args']['dataset_type']

        # Load algorithm metrics.
        exp_metrics_path = plots_folder_path + exp_id + '/scalar_metrics.json'
        with open(exp_metrics_path, 'r') as f:
            metrics_data = json.load(f)
        f.close()

        row_data['qvals_avg_error'] = metrics_data['qvals_avg_error']
        row_data['qvals_summed_error'] = metrics_data['qvals_summed_error']
        row_data['rollouts_rewards_final'] = metrics_data['rollouts_rewards_final']
        row_data['force_dataset_coverage'] = args['dataset_args']['force_full_coverage']

        if row_data['dataset_type_id'] == 'dirichlet':
            dataset_type_arg = f"(alpha={args['dataset_args']['dirichlet_dataset_args']['dirichlet_alpha_coef']},force_coverage={args['dataset_args']['force_full_coverage']})"
        elif row_data['dataset_type_id'] == 'eps-greedy':
            dataset_type_arg = f"(epsilon={args['dataset_args']['eps_greedy_dataset_args']['epsilon']},force_coverage={args['dataset_args']['force_full_coverage']})"
        elif row_data['dataset_type_id'] == 'boltzmann':
            dataset_type_arg = f"(temperature={args['dataset_args']['boltzmann_dataset_args']['temperature']},force_coverage={args['dataset_args']['force_full_coverage']})"
        else:
            raise ValueError('Unknown dataset type.')

        # Load dataset metrics.
        exp_folder_path = data_folder_path + exp_id + '.tar.gz'
        tar = tarfile.open(exp_folder_path)
        data_file = tar.extractfile("{0}/dataset_info.json".format(exp_id))
        dataset_info = json.load(data_file)
        dataset_info = json.loads(dataset_info)

        row_data['dataset_entropy'] = dataset_info['dataset_entropy']

        if "dataset_sa_counts" in list(dataset_info.keys()):
            dataset_sa_counts = np.array(dataset_info['dataset_sa_counts'])
        else:
            dataset_sa_counts = np.array(dataset_info['sa_counts'])
        dataset_sa_counts = dataset_sa_counts.flatten()

        num_non_zeros = (dataset_sa_counts != 0).sum()
        row_data['dataset_coverage'] = num_non_zeros / len(dataset_sa_counts)

        row_data['kl_dist'] = np.min([stats.entropy(optimal_dist, np.array(dataset_info['dataset_dist']) + 1e-06)
                        for optimal_dist in opt_sampling_dists[args['env_name']]])
        row_data['chi_dist'] = np.min([chi_div(optimal_dist, np.array(dataset_info['dataset_dist']))
                        for optimal_dist in opt_sampling_dists[args['env_name']]])

        info_text = f"Exp-id: {row_data['id']}<br>Environment: {row_data['env_id']}<br>Algorithm: {row_data['algo_id']}<br>Dataset type: {row_data['dataset_type_id']}{dataset_type_arg}<br>Dataset coverage: {row_data['dataset_coverage']:.2f}<br>Dataset entropy: {row_data['dataset_entropy']:.2f}<br>Q-values avg error: {row_data['qvals_avg_error']:.2f}<br>Q-values summed error: {row_data['qvals_summed_error']:.2f}<br>Rollouts rewards: {row_data['rollouts_rewards_final']:.2f}<br>KL-dist: {row_data['kl_dist']:.2f}"
        row_data['info_text'] = info_text

        df_rows.append(row_data)

    print('Finished parsing data.')

    df = pd.DataFrame(df_rows)
    df.to_csv(DATA_FOLDER_PATH_1 + 'parsed_data.csv')

if __name__ == "__main__":
    data_to_csv(EXP_IDS)
