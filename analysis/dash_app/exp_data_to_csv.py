from multiprocessing.sharedctypes import Value
import os
import json
import pathlib
import tarfile

import numpy as np
import pandas as pd

# Path to folder containing data files.
DATA_FOLDER_PATH_1 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH_1 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/plots/'

# Path to folder containing data files (second folder).
DATA_FOLDER_PATH_2 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/data/ilu_server/'
PLOTS_FOLDER_PATH_2 = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/plots/ilu_server/'

EXP_IDS = [
# gridEnv1. (ACME branch data).
# DQN.
# Dirichlet.
'gridEnv1_offline_dqn_2022-02-06-00-50-49', 'gridEnv1_offline_dqn_2022-02-06-00-57-11', 'gridEnv1_offline_dqn_2022-02-06-01-03-31', 'gridEnv1_offline_dqn_2022-02-06-01-09-52', 'gridEnv1_offline_dqn_2022-02-06-01-16-15', 'gridEnv1_offline_dqn_2022-02-06-01-22-34', 'gridEnv1_offline_dqn_2022-02-06-01-28-57', 'gridEnv1_offline_dqn_2022-02-06-01-35-41', 'gridEnv1_offline_dqn_2022-02-06-01-42-03', 'gridEnv1_offline_dqn_2022-02-06-01-48-50', 'gridEnv1_offline_dqn_2022-02-06-01-55-12', 'gridEnv1_offline_dqn_2022-02-06-02-01-35', 'gridEnv1_offline_dqn_2022-02-06-02-08-19', 'gridEnv1_offline_dqn_2022-02-06-02-14-43',
# Eps-greedy.
'gridEnv1_offline_dqn_2022-02-06-02-21-03', 'gridEnv1_offline_dqn_2022-02-06-02-27-22', 'gridEnv1_offline_dqn_2022-02-06-02-33-41', 'gridEnv1_offline_dqn_2022-02-06-02-40-01', 'gridEnv1_offline_dqn_2022-02-06-02-46-20', 'gridEnv1_offline_dqn_2022-02-06-02-52-37', 'gridEnv1_offline_dqn_2022-02-06-02-58-56', 'gridEnv1_offline_dqn_2022-02-06-03-05-14', 'gridEnv1_offline_dqn_2022-02-06-03-11-35', 'gridEnv1_offline_dqn_2022-02-06-03-18-17', 'gridEnv1_offline_dqn_2022-02-06-03-24-38', 'gridEnv1_offline_dqn_2022-02-06-03-30-56', 'gridEnv1_offline_dqn_2022-02-06-03-37-16', 'gridEnv1_offline_dqn_2022-02-06-03-43-34', 'gridEnv1_offline_dqn_2022-02-06-03-49-52', 'gridEnv1_offline_dqn_2022-02-06-03-56-12', 'gridEnv1_offline_dqn_2022-02-06-04-02-54', 'gridEnv1_offline_dqn_2022-02-06-04-09-16', 'gridEnv1_offline_dqn_2022-02-06-04-15-36', 'gridEnv1_offline_dqn_2022-02-06-04-21-53', 'gridEnv1_offline_dqn_2022-02-06-04-28-13', 'gridEnv1_offline_dqn_2022-02-06-04-34-58',
# Boltzmann.
'gridEnv1_offline_dqn_2022-02-06-04-41-18', 'gridEnv1_offline_dqn_2022-02-06-04-47-39', 'gridEnv1_offline_dqn_2022-02-06-04-53-59', 'gridEnv1_offline_dqn_2022-02-06-05-00-20', 'gridEnv1_offline_dqn_2022-02-06-05-07-03', 'gridEnv1_offline_dqn_2022-02-06-05-13-21', 'gridEnv1_offline_dqn_2022-02-06-05-19-42', 'gridEnv1_offline_dqn_2022-02-06-05-26-04', 'gridEnv1_offline_dqn_2022-02-06-05-32-24', 'gridEnv1_offline_dqn_2022-02-06-05-38-43', 'gridEnv1_offline_dqn_2022-02-06-05-45-03', 'gridEnv1_offline_dqn_2022-02-06-05-51-22', 'gridEnv1_offline_dqn_2022-02-06-05-57-43', 'gridEnv1_offline_dqn_2022-02-06-06-04-04', 'gridEnv1_offline_dqn_2022-02-06-06-10-47', 'gridEnv1_offline_dqn_2022-02-06-06-17-35', 'gridEnv1_offline_dqn_2022-02-06-06-24-18', 'gridEnv1_offline_dqn_2022-02-06-06-30-39', 'gridEnv1_offline_dqn_2022-02-06-06-36-57', 'gridEnv1_offline_dqn_2022-02-06-06-43-16', 'gridEnv1_offline_dqn_2022-02-06-06-49-37', 'gridEnv1_offline_dqn_2022-02-06-06-55-58', 'gridEnv1_offline_dqn_2022-02-06-07-02-16', 'gridEnv1_offline_dqn_2022-02-06-07-08-38', 'gridEnv1_offline_dqn_2022-02-06-07-14-58', 'gridEnv1_offline_dqn_2022-02-06-07-21-18', 'gridEnv1_offline_dqn_2022-02-06-07-27-38', 'gridEnv1_offline_dqn_2022-02-06-07-34-00', 'gridEnv1_offline_dqn_2022-02-06-07-40-20', 'gridEnv1_offline_dqn_2022-02-06-07-46-44', 'gridEnv1_offline_dqn_2022-02-06-07-53-06', 'gridEnv1_offline_dqn_2022-02-06-07-59-27', 'gridEnv1_offline_dqn_2022-02-06-08-06-10', 'gridEnv1_offline_dqn_2022-02-06-08-12-53', 'gridEnv1_offline_dqn_2022-02-06-08-19-36', 'gridEnv1_offline_dqn_2022-02-06-08-25-56', 'gridEnv1_offline_dqn_2022-02-06-08-32-41', 'gridEnv1_offline_dqn_2022-02-06-08-39-01', 'gridEnv1_offline_dqn_2022-02-06-08-45-21', 'gridEnv1_offline_dqn_2022-02-06-08-52-04', 'gridEnv1_offline_dqn_2022-02-06-08-58-22', 'gridEnv1_offline_dqn_2022-02-06-09-04-42',

# CQL.
# Dirichlet.

# Eps-greedy.

# Boltzmann.

# gridEnv2.
# DQN.
# Dirichlet.
'gridEnv2_offline_dqn_2022-01-22-14-07-39', 'gridEnv2_offline_dqn_2022-01-22-14-14-53', 'gridEnv2_offline_dqn_2022-01-22-14-22-02', 'gridEnv2_offline_dqn_2022-01-22-14-29-02', 'gridEnv2_offline_dqn_2022-01-22-14-36-03', 'gridEnv2_offline_dqn_2022-01-22-14-42-59', 'gridEnv2_offline_dqn_2022-01-22-14-49-54', 'gridEnv2_offline_dqn_2022-01-22-14-56-55', 'gridEnv2_offline_dqn_2022-01-22-15-03-58', 'gridEnv2_offline_dqn_2022-01-22-15-10-54', 'gridEnv2_offline_dqn_2022-01-22-15-17-52', 'gridEnv2_offline_dqn_2022-01-22-15-24-45', 'gridEnv2_offline_dqn_2022-01-22-15-31-45', 'gridEnv2_offline_dqn_2022-01-22-15-38-44',
# Eps-greedy.
'gridEnv2_offline_dqn_2022-01-22-15-45-43', 'gridEnv2_offline_dqn_2022-01-22-15-52-33', 'gridEnv2_offline_dqn_2022-01-22-15-59-38', 'gridEnv2_offline_dqn_2022-01-22-16-06-28', 'gridEnv2_offline_dqn_2022-01-22-16-13-25', 'gridEnv2_offline_dqn_2022-01-22-16-20-14', 'gridEnv2_offline_dqn_2022-01-22-16-27-16', 'gridEnv2_offline_dqn_2022-01-22-16-34-09', 'gridEnv2_offline_dqn_2022-01-22-16-41-12', 'gridEnv2_offline_dqn_2022-01-22-16-48-13', 'gridEnv2_offline_dqn_2022-01-22-16-55-09', 'gridEnv2_offline_dqn_2022-01-22-17-02-03', 'gridEnv2_offline_dqn_2022-01-22-17-08-56', 'gridEnv2_offline_dqn_2022-01-22-17-15-51', 'gridEnv2_offline_dqn_2022-01-22-17-22-52', 'gridEnv2_offline_dqn_2022-01-22-17-29-50', 'gridEnv2_offline_dqn_2022-01-22-17-36-53', 'gridEnv2_offline_dqn_2022-01-22-17-43-47', 'gridEnv2_offline_dqn_2022-01-22-17-50-42', 'gridEnv2_offline_dqn_2022-01-22-17-57-44', 'gridEnv2_offline_dqn_2022-01-22-18-04-37', 'gridEnv2_offline_dqn_2022-01-22-18-13-09',
# Boltzmann.
'gridEnv2_offline_dqn_2022-01-22-18-20-22', 'gridEnv2_offline_dqn_2022-01-22-18-27-28', 'gridEnv2_offline_dqn_2022-01-22-18-34-28', 'gridEnv2_offline_dqn_2022-01-22-18-41-24', 'gridEnv2_offline_dqn_2022-01-22-18-48-29', 'gridEnv2_offline_dqn_2022-01-22-18-55-22', 'gridEnv2_offline_dqn_2022-01-22-19-02-20', 'gridEnv2_offline_dqn_2022-01-22-19-09-19', 'gridEnv2_offline_dqn_2022-01-22-19-16-23', 'gridEnv2_offline_dqn_2022-01-22-19-23-29', 'gridEnv2_offline_dqn_2022-01-22-19-30-34', 'gridEnv2_offline_dqn_2022-01-22-19-37-33', 'gridEnv2_offline_dqn_2022-01-22-19-44-34', 'gridEnv2_offline_dqn_2022-01-22-19-51-30', 'gridEnv2_offline_dqn_2022-01-22-19-58-28', 'gridEnv2_offline_dqn_2022-01-22-20-05-26', 'gridEnv2_offline_dqn_2022-01-22-20-12-28', 'gridEnv2_offline_dqn_2022-01-22-20-19-34', 'gridEnv2_offline_dqn_2022-01-22-20-26-54', 'gridEnv2_offline_dqn_2022-01-22-20-34-06', 'gridEnv2_offline_dqn_2022-01-22-20-41-35', 'gridEnv2_offline_dqn_2022-01-22-20-48-51', 'gridEnv2_offline_dqn_2022-01-22-20-55-58', 'gridEnv2_offline_dqn_2022-01-22-21-02-58', 'gridEnv2_offline_dqn_2022-01-22-21-10-02', 'gridEnv2_offline_dqn_2022-01-22-21-17-11', 'gridEnv2_offline_dqn_2022-01-22-21-24-15', 'gridEnv2_offline_dqn_2022-01-22-21-31-21', 'gridEnv2_offline_dqn_2022-01-22-21-38-22', 'gridEnv2_offline_dqn_2022-01-22-21-45-34', 'gridEnv2_offline_dqn_2022-01-22-21-52-37', 'gridEnv2_offline_dqn_2022-01-22-21-59-43', 'gridEnv2_offline_dqn_2022-01-22-22-06-48', 'gridEnv2_offline_dqn_2022-01-22-22-13-47', 'gridEnv2_offline_dqn_2022-01-22-22-20-53', 'gridEnv2_offline_dqn_2022-01-22-22-27-57', 'gridEnv2_offline_dqn_2022-01-22-22-34-54', 'gridEnv2_offline_dqn_2022-01-22-22-41-52', 'gridEnv2_offline_dqn_2022-01-22-22-48-51', 'gridEnv2_offline_dqn_2022-01-22-22-55-57', 'gridEnv2_offline_dqn_2022-01-22-23-03-03', 'gridEnv2_offline_dqn_2022-01-22-23-09-59',

# CQL.
# Dirichlet.
'gridEnv2_offline_cql_2022-01-23-01-26-31', 'gridEnv2_offline_cql_2022-01-23-01-36-47', 'gridEnv2_offline_cql_2022-01-23-01-46-30', 'gridEnv2_offline_cql_2022-01-23-01-56-08', 'gridEnv2_offline_cql_2022-01-23-02-05-56', 'gridEnv2_offline_cql_2022-01-23-02-15-41', 'gridEnv2_offline_cql_2022-01-23-02-25-30', 'gridEnv2_offline_cql_2022-01-23-02-35-19', 'gridEnv2_offline_cql_2022-01-23-02-45-08', 'gridEnv2_offline_cql_2022-01-23-02-54-59', 'gridEnv2_offline_cql_2022-01-23-03-04-38', 'gridEnv2_offline_cql_2022-01-23-03-14-25', 'gridEnv2_offline_cql_2022-01-23-03-24-07', 'gridEnv2_offline_cql_2022-01-23-03-33-55',
# Eps-greedy.
'gridEnv2_offline_cql_2022-01-23-03-43-36', 'gridEnv2_offline_cql_2022-01-23-03-53-20', 'gridEnv2_offline_cql_2022-01-23-04-03-08', 'gridEnv2_offline_cql_2022-01-23-04-12-46', 'gridEnv2_offline_cql_2022-01-23-04-22-26', 'gridEnv2_offline_cql_2022-01-23-04-32-07', 'gridEnv2_offline_cql_2022-01-23-04-41-53', 'gridEnv2_offline_cql_2022-01-23-04-51-45', 'gridEnv2_offline_cql_2022-01-23-05-01-36', 'gridEnv2_offline_cql_2022-01-23-05-11-30', 'gridEnv2_offline_cql_2022-01-23-05-21-11', 'gridEnv2_offline_cql_2022-01-23-05-30-55', 'gridEnv2_offline_cql_2022-01-23-05-40-45', 'gridEnv2_offline_cql_2022-01-23-05-50-24', 'gridEnv2_offline_cql_2022-01-23-06-00-03', 'gridEnv2_offline_cql_2022-01-23-06-09-55', 'gridEnv2_offline_cql_2022-01-23-06-19-39', 'gridEnv2_offline_cql_2022-01-23-06-29-16', 'gridEnv2_offline_cql_2022-01-23-06-38-59', 'gridEnv2_offline_cql_2022-01-23-06-48-34', 'gridEnv2_offline_cql_2022-01-23-06-58-15', 'gridEnv2_offline_cql_2022-01-23-07-07-58',
# Boltzmann.
'gridEnv2_offline_cql_2022-01-23-07-17-40', 'gridEnv2_offline_cql_2022-01-23-07-27-29', 'gridEnv2_offline_cql_2022-01-23-07-37-07', 'gridEnv2_offline_cql_2022-01-23-07-46-51', 'gridEnv2_offline_cql_2022-01-23-07-56-34', 'gridEnv2_offline_cql_2022-01-23-08-06-12', 'gridEnv2_offline_cql_2022-01-23-08-16-11', 'gridEnv2_offline_cql_2022-01-23-08-25-52', 'gridEnv2_offline_cql_2022-01-23-08-35-39', 'gridEnv2_offline_cql_2022-01-23-08-45-15', 'gridEnv2_offline_cql_2022-01-23-08-55-06', 'gridEnv2_offline_cql_2022-01-23-09-04-48', 'gridEnv2_offline_cql_2022-01-23-09-14-33', 'gridEnv2_offline_cql_2022-01-23-09-24-13', 'gridEnv2_offline_cql_2022-01-23-09-34-03', 'gridEnv2_offline_cql_2022-01-23-09-43-42', 'gridEnv2_offline_cql_2022-01-23-09-53-25', 'gridEnv2_offline_cql_2022-01-23-10-03-03', 'gridEnv2_offline_cql_2022-01-23-10-12-51', 'gridEnv2_offline_cql_2022-01-23-10-22-30', 'gridEnv2_offline_cql_2022-01-23-10-32-11', 'gridEnv2_offline_cql_2022-01-23-10-41-50', 'gridEnv2_offline_cql_2022-01-23-10-51-34', 'gridEnv2_offline_cql_2022-01-23-11-01-26', 'gridEnv2_offline_cql_2022-01-23-11-11-12', 'gridEnv2_offline_cql_2022-01-23-11-21-05', 'gridEnv2_offline_cql_2022-01-23-11-30-48', 'gridEnv2_offline_cql_2022-01-23-11-40-32', 'gridEnv2_offline_cql_2022-01-23-11-50-15', 'gridEnv2_offline_cql_2022-01-23-12-00-11', 'gridEnv2_offline_cql_2022-01-23-12-10-17', 'gridEnv2_offline_cql_2022-01-23-12-20-14', 'gridEnv2_offline_cql_2022-01-23-12-30-21', 'gridEnv2_offline_cql_2022-01-23-12-40-27', 'gridEnv2_offline_cql_2022-01-23-12-50-45', 'gridEnv2_offline_cql_2022-01-23-13-00-52', 'gridEnv2_offline_cql_2022-01-23-13-10-55', 'gridEnv2_offline_cql_2022-01-23-13-20-58', 'gridEnv2_offline_cql_2022-01-23-13-30-57', 'gridEnv2_offline_cql_2022-01-23-13-41-00', 'gridEnv2_offline_cql_2022-01-23-13-51-06', 'gridEnv2_offline_cql_2022-01-23-14-01-10',

# multiPathEnv.
# DQN.
# Dirichlet.
'multiPathEnv_offline_dqn_2022-01-24-01-35-30', 'multiPathEnv_offline_dqn_2022-01-24-01-44-13', 'multiPathEnv_offline_dqn_2022-01-24-01-51-04', 'multiPathEnv_offline_dqn_2022-01-24-01-57-49', 'multiPathEnv_offline_dqn_2022-01-24-02-04-37', 'multiPathEnv_offline_dqn_2022-01-24-02-11-24', 'multiPathEnv_offline_dqn_2022-01-24-02-18-17', 'multiPathEnv_offline_dqn_2022-01-24-02-25-16', 'multiPathEnv_offline_dqn_2022-01-24-02-32-08', 'multiPathEnv_offline_dqn_2022-01-24-02-38-56', 'multiPathEnv_offline_dqn_2022-01-24-02-45-47', 'multiPathEnv_offline_dqn_2022-01-24-02-52-37', 'multiPathEnv_offline_dqn_2022-01-24-02-59-25', 'multiPathEnv_offline_dqn_2022-01-24-03-06-12',
# Eps-greedy.
'multiPathEnv_offline_dqn_2022-01-24-03-12-59', 'multiPathEnv_offline_dqn_2022-01-24-03-19-54', 'multiPathEnv_offline_dqn_2022-01-24-03-26-42', 'multiPathEnv_offline_dqn_2022-01-24-03-33-28', 'multiPathEnv_offline_dqn_2022-01-24-03-40-14', 'multiPathEnv_offline_dqn_2022-01-24-03-47-09', 'multiPathEnv_offline_dqn_2022-01-24-03-53-58', 'multiPathEnv_offline_dqn_2022-01-24-04-00-51', 'multiPathEnv_offline_dqn_2022-01-24-04-07-37', 'multiPathEnv_offline_dqn_2022-01-24-04-14-26', 'multiPathEnv_offline_dqn_2022-01-24-04-21-20', 'multiPathEnv_offline_dqn_2022-01-24-04-28-18', 'multiPathEnv_offline_dqn_2022-01-24-04-35-06', 'multiPathEnv_offline_dqn_2022-01-24-04-41-55', 'multiPathEnv_offline_dqn_2022-01-24-04-48-49', 'multiPathEnv_offline_dqn_2022-01-24-04-55-34', 'multiPathEnv_offline_dqn_2022-01-24-05-02-21', 'multiPathEnv_offline_dqn_2022-01-24-05-09-11', 'multiPathEnv_offline_dqn_2022-01-24-05-15-59', 'multiPathEnv_offline_dqn_2022-01-24-05-22-54', 'multiPathEnv_offline_dqn_2022-01-24-05-29-42', 'multiPathEnv_offline_dqn_2022-01-24-05-36-33',
# Boltzmann.
'multiPathEnv_offline_dqn_2022-01-24-05-43-24', 'multiPathEnv_offline_dqn_2022-01-24-05-50-15', 'multiPathEnv_offline_dqn_2022-01-24-05-57-05', 'multiPathEnv_offline_dqn_2022-01-24-06-04-01', 'multiPathEnv_offline_dqn_2022-01-24-06-10-50', 'multiPathEnv_offline_dqn_2022-01-24-06-17-40', 'multiPathEnv_offline_dqn_2022-01-24-06-24-36', 'multiPathEnv_offline_dqn_2022-01-24-06-31-25', 'multiPathEnv_offline_dqn_2022-01-24-06-38-21', 'multiPathEnv_offline_dqn_2022-01-24-06-45-21', 'multiPathEnv_offline_dqn_2022-01-24-06-52-14', 'multiPathEnv_offline_dqn_2022-01-24-06-59-08', 'multiPathEnv_offline_dqn_2022-01-24-07-06-09', 'multiPathEnv_offline_dqn_2022-01-24-07-12-54', 'multiPathEnv_offline_dqn_2022-01-24-07-19-48', 'multiPathEnv_offline_dqn_2022-01-24-07-26-37', 'multiPathEnv_offline_dqn_2022-01-24-07-33-24', 'multiPathEnv_offline_dqn_2022-01-24-07-40-17', 'multiPathEnv_offline_dqn_2022-01-24-07-47-11', 'multiPathEnv_offline_dqn_2022-01-24-07-54-06', 'multiPathEnv_offline_dqn_2022-01-24-08-00-53', 'multiPathEnv_offline_dqn_2022-01-24-08-07-42', 'multiPathEnv_offline_dqn_2022-01-24-08-14-32', 'multiPathEnv_offline_dqn_2022-01-24-08-21-21', 'multiPathEnv_offline_dqn_2022-01-24-08-28-11', 'multiPathEnv_offline_dqn_2022-01-24-08-35-05', 'multiPathEnv_offline_dqn_2022-01-24-08-41-58', 'multiPathEnv_offline_dqn_2022-01-24-08-48-51', 'multiPathEnv_offline_dqn_2022-01-24-08-55-44', 'multiPathEnv_offline_dqn_2022-01-24-09-02-34', 'multiPathEnv_offline_dqn_2022-01-24-09-09-28', 'multiPathEnv_offline_dqn_2022-01-24-09-16-23', 'multiPathEnv_offline_dqn_2022-01-24-09-23-20', 'multiPathEnv_offline_dqn_2022-01-24-09-30-08', 'multiPathEnv_offline_dqn_2022-01-24-09-36-58', 'multiPathEnv_offline_dqn_2022-01-24-09-43-51', 'multiPathEnv_offline_dqn_2022-01-24-09-50-41', 'multiPathEnv_offline_dqn_2022-01-24-09-57-35', 'multiPathEnv_offline_dqn_2022-01-24-10-04-29', 'multiPathEnv_offline_dqn_2022-01-24-10-11-20', 'multiPathEnv_offline_dqn_2022-01-24-10-18-15', 'multiPathEnv_offline_dqn_2022-01-24-10-25-04',

# CQL.
# Dirichlet.
'multiPathEnv_offline_cql_2022-01-24-10-32-06', 'multiPathEnv_offline_cql_2022-01-24-10-41-55', 'multiPathEnv_offline_cql_2022-01-24-10-51-30', 'multiPathEnv_offline_cql_2022-01-24-11-01-12', 'multiPathEnv_offline_cql_2022-01-24-11-10-57', 'multiPathEnv_offline_cql_2022-01-24-11-20-35', 'multiPathEnv_offline_cql_2022-01-24-11-30-13', 'multiPathEnv_offline_cql_2022-01-24-11-39-55', 'multiPathEnv_offline_cql_2022-01-24-11-49-34', 'multiPathEnv_offline_cql_2022-01-24-11-59-06', 'multiPathEnv_offline_cql_2022-01-24-12-09-00', 'multiPathEnv_offline_cql_2022-01-24-12-18-56', 'multiPathEnv_offline_cql_2022-01-24-12-28-58', 'multiPathEnv_offline_cql_2022-01-24-12-38-56',
# Eps-greedy.
'multiPathEnv_offline_cql_2022-01-24-12-48-53', 'multiPathEnv_offline_cql_2022-01-24-12-58-58', 'multiPathEnv_offline_cql_2022-01-24-13-09-01', 'multiPathEnv_offline_cql_2022-01-24-13-18-55', 'multiPathEnv_offline_cql_2022-01-24-13-28-50', 'multiPathEnv_offline_cql_2022-01-24-13-38-45', 'multiPathEnv_offline_cql_2022-01-24-13-48-41', 'multiPathEnv_offline_cql_2022-01-24-13-58-36', 'multiPathEnv_offline_cql_2022-01-24-14-08-34', 'multiPathEnv_offline_cql_2022-01-24-14-18-28', 'multiPathEnv_offline_cql_2022-01-24-14-28-21', 'multiPathEnv_offline_cql_2022-01-24-14-38-11', 'multiPathEnv_offline_cql_2022-01-24-14-48-01', 'multiPathEnv_offline_cql_2022-01-24-14-57-56', 'multiPathEnv_offline_cql_2022-01-24-15-07-50', 'multiPathEnv_offline_cql_2022-01-24-15-17-40', 'multiPathEnv_offline_cql_2022-01-24-15-28-07', 'multiPathEnv_offline_cql_2022-01-24-15-37-52', 'multiPathEnv_offline_cql_2022-01-24-15-47-42', 'multiPathEnv_offline_cql_2022-01-24-15-57-25', 'multiPathEnv_offline_cql_2022-01-24-16-06-59', 'multiPathEnv_offline_cql_2022-01-24-16-16-28',
# Boltzmann.
'multiPathEnv_offline_cql_2022-01-24-16-26-01', 'multiPathEnv_offline_cql_2022-01-24-16-35-34', 'multiPathEnv_offline_cql_2022-01-24-16-45-07', 'multiPathEnv_offline_cql_2022-01-24-16-54-40', 'multiPathEnv_offline_cql_2022-01-24-17-04-08', 'multiPathEnv_offline_cql_2022-01-24-17-13-48', 'multiPathEnv_offline_cql_2022-01-24-17-23-23', 'multiPathEnv_offline_cql_2022-01-24-17-33-04', 'multiPathEnv_offline_cql_2022-01-24-17-43-00', 'multiPathEnv_offline_cql_2022-01-24-17-52-39', 'multiPathEnv_offline_cql_2022-01-24-18-02-38', 'multiPathEnv_offline_cql_2022-01-24-18-12-32', 'multiPathEnv_offline_cql_2022-01-24-18-22-36', 'multiPathEnv_offline_cql_2022-01-24-18-32-15', 'multiPathEnv_offline_cql_2022-01-24-18-41-51', 'multiPathEnv_offline_cql_2022-01-24-18-51-45', 'multiPathEnv_offline_cql_2022-01-24-19-01-40', 'multiPathEnv_offline_cql_2022-01-24-19-11-28', 'multiPathEnv_offline_cql_2022-01-24-19-21-28', 'multiPathEnv_offline_cql_2022-01-24-19-31-11', 'multiPathEnv_offline_cql_2022-01-24-19-40-54', 'multiPathEnv_offline_cql_2022-01-24-19-50-36', 'multiPathEnv_offline_cql_2022-01-24-20-00-22', 'multiPathEnv_offline_cql_2022-01-24-20-09-57', 'multiPathEnv_offline_cql_2022-01-24-20-20-09', 'multiPathEnv_offline_cql_2022-01-24-20-29-56', 'multiPathEnv_offline_cql_2022-01-24-20-39-46', 'multiPathEnv_offline_cql_2022-01-24-20-49-15', 'multiPathEnv_offline_cql_2022-01-24-20-58-56', 'multiPathEnv_offline_cql_2022-01-24-21-08-51', 'multiPathEnv_offline_cql_2022-01-24-21-18-51', 'multiPathEnv_offline_cql_2022-01-24-21-28-22', 'multiPathEnv_offline_cql_2022-01-24-21-38-12', 'multiPathEnv_offline_cql_2022-01-24-21-47-43', 'multiPathEnv_offline_cql_2022-01-24-21-57-27', 'multiPathEnv_offline_cql_2022-01-24-22-06-58', 'multiPathEnv_offline_cql_2022-01-24-22-16-31', 'multiPathEnv_offline_cql_2022-01-24-22-25-59', 'multiPathEnv_offline_cql_2022-01-24-22-35-35', 'multiPathEnv_offline_cql_2022-01-24-22-45-00', 'multiPathEnv_offline_cql_2022-01-24-22-54-52', 'multiPathEnv_offline_cql_2022-01-24-23-04-32',

# mountaincar (ACME branch data).
# DQN.
# Dirichlet.
'mountaincar_offline_dqn_2022-02-02-21-38-00', 'mountaincar_offline_dqn_2022-02-02-21-52-33', 'mountaincar_offline_dqn_2022-02-02-22-07-09', 'mountaincar_offline_dqn_2022-02-02-22-21-44', 'mountaincar_offline_dqn_2022-02-02-22-36-15', 'mountaincar_offline_dqn_2022-02-02-22-50-50', 'mountaincar_offline_dqn_2022-02-02-23-05-29', 'mountaincar_offline_dqn_2022-02-02-23-19-59', 'mountaincar_offline_dqn_2022-02-02-23-34-35', 'mountaincar_offline_dqn_2022-02-02-23-49-09', 'mountaincar_offline_dqn_2022-02-03-00-03-43', 'mountaincar_offline_dqn_2022-02-03-00-18-20', 'mountaincar_offline_dqn_2022-02-03-00-32-53', 'mountaincar_offline_dqn_2022-02-03-00-47-28',
# Eps-greedy.
'mountaincar_offline_dqn_2022-02-03-10-50-39', 'mountaincar_offline_dqn_2022-02-03-11-04-59', 'mountaincar_offline_dqn_2022-02-03-11-19-33', 'mountaincar_offline_dqn_2022-02-03-11-33-43', 'mountaincar_offline_dqn_2022-02-03-11-48-04', 'mountaincar_offline_dqn_2022-02-03-12-02-31', 'mountaincar_offline_dqn_2022-02-03-12-16-45', 'mountaincar_offline_dqn_2022-02-03-12-31-02', 'mountaincar_offline_dqn_2022-02-03-12-45-18', 'mountaincar_offline_dqn_2022-02-03-12-59-39', 'mountaincar_offline_dqn_2022-02-03-13-13-58', 'mountaincar_offline_dqn_2022-02-03-13-28-16', 'mountaincar_offline_dqn_2022-02-03-13-42-43', 'mountaincar_offline_dqn_2022-02-03-13-56-55', 'mountaincar_offline_dqn_2022-02-03-14-11-12', 'mountaincar_offline_dqn_2022-02-03-14-25-25', 'mountaincar_offline_dqn_2022-02-03-14-39-38', 'mountaincar_offline_dqn_2022-02-03-14-53-53', 'mountaincar_offline_dqn_2022-02-03-15-08-11', 'mountaincar_offline_dqn_2022-02-03-15-22-26', 'mountaincar_offline_dqn_2022-02-03-15-36-44',
# Boltzmann.
'mountaincar_offline_dqn_2022-02-03-15-51-02', 'mountaincar_offline_dqn_2022-02-03-16-05-18', 'mountaincar_offline_dqn_2022-02-03-16-19-45', 'mountaincar_offline_dqn_2022-02-03-16-34-09', 'mountaincar_offline_dqn_2022-02-03-16-48-34', 'mountaincar_offline_dqn_2022-02-03-17-03-42', 'mountaincar_offline_dqn_2022-02-03-17-18-56', 'mountaincar_offline_dqn_2022-02-03-17-34-08', 'mountaincar_offline_dqn_2022-02-03-17-49-47', 'mountaincar_offline_dqn_2022-02-03-18-04-53', 'mountaincar_offline_dqn_2022-02-03-18-20-00', 'mountaincar_offline_dqn_2022-02-03-18-35-14', 'mountaincar_offline_dqn_2022-02-03-18-50-22', 'mountaincar_offline_dqn_2022-02-03-19-05-26', 'mountaincar_offline_dqn_2022-02-03-19-20-25', 'mountaincar_offline_dqn_2022-02-03-19-35-27', 'mountaincar_offline_dqn_2022-02-03-19-50-31', 'mountaincar_offline_dqn_2022-02-03-20-05-38', 'mountaincar_offline_dqn_2022-02-03-20-20-40', 'mountaincar_offline_dqn_2022-02-03-20-35-43', 'mountaincar_offline_dqn_2022-02-03-20-50-49', 'mountaincar_offline_dqn_2022-02-03-21-05-56', 'mountaincar_offline_dqn_2022-02-03-21-21-00', 'mountaincar_offline_dqn_2022-02-03-21-36-12', 'mountaincar_offline_dqn_2022-02-03-21-51-18', 'mountaincar_offline_dqn_2022-02-03-22-06-47', 'mountaincar_offline_dqn_2022-02-03-22-22-13', 'mountaincar_offline_dqn_2022-02-03-22-37-24', 'mountaincar_offline_dqn_2022-02-03-22-52-32', 'mountaincar_offline_dqn_2022-02-03-23-07-39', 'mountaincar_offline_dqn_2022-02-03-23-22-45', 'mountaincar_offline_dqn_2022-02-03-23-37-57', 'mountaincar_offline_dqn_2022-02-03-23-53-07', 'mountaincar_offline_dqn_2022-02-04-00-08-15', 'mountaincar_offline_dqn_2022-02-04-00-23-23', 'mountaincar_offline_dqn_2022-02-04-00-38-28', 'mountaincar_offline_dqn_2022-02-04-00-53-53', 'mountaincar_offline_dqn_2022-02-04-01-09-18', 'mountaincar_offline_dqn_2022-02-04-01-24-22', 'mountaincar_offline_dqn_2022-02-04-01-39-47', 'mountaincar_offline_dqn_2022-02-04-01-54-55', 'mountaincar_offline_dqn_2022-02-04-02-09-58', 'mountaincar_offline_dqn_2022-02-04-02-24-59',
]


def data_to_csv(exp_ids):
    print('Parsing data to csv file.')

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

        info_text = f"Exp-id: {row_data['id']}<br>Environment: {row_data['env_id']}<br>Algorithm: {row_data['algo_id']}<br>Dataset type: {row_data['dataset_type_id']}{dataset_type_arg}<br>Dataset coverage: {row_data['dataset_coverage']:.2f}<br>Dataset entropy: {row_data['dataset_entropy']:.2f}<br>Q-values avg error: {row_data['qvals_avg_error']:.2f}<br>Q-values summed error: {row_data['qvals_summed_error']:.2f}<br>Rollouts rewards: {row_data['rollouts_rewards_final']:.2f}"
        row_data['info_text'] = info_text

        df_rows.append(row_data)

    print('Finished parsing data.')

    df = pd.DataFrame(df_rows)
    df.to_csv(DATA_FOLDER_PATH_1 + 'parsed_data.csv')

if __name__ == "__main__":
    data_to_csv(EXP_IDS)
