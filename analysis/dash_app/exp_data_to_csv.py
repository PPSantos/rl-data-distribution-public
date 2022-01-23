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
# gridEnv1.
# DQN.
# Dirichlet.
'gridEnv1_offline_dqn_2022-01-18-22-07-14', 'gridEnv1_offline_dqn_2022-01-18-22-15-26', 'gridEnv1_offline_dqn_2022-01-18-22-23-08', 'gridEnv1_offline_dqn_2022-01-18-22-31-01', 'gridEnv1_offline_dqn_2022-01-18-22-38-56', 'gridEnv1_offline_dqn_2022-01-18-22-46-39', 'gridEnv1_offline_dqn_2022-01-18-22-54-30', 'gridEnv1_offline_dqn_2022-01-21-21-47-49', 'gridEnv1_offline_dqn_2022-01-21-21-55-07', 'gridEnv1_offline_dqn_2022-01-21-22-02-08', 'gridEnv1_offline_dqn_2022-01-21-22-09-10', 'gridEnv1_offline_dqn_2022-01-21-22-16-09', 'gridEnv1_offline_dqn_2022-01-21-22-23-09', 'gridEnv1_offline_dqn_2022-01-21-22-30-09',
# Eps-greedy.
'gridEnv1_offline_dqn_2022-01-19-16-05-47', 'gridEnv1_offline_dqn_2022-01-19-16-14-01', 'gridEnv1_offline_dqn_2022-01-19-16-21-57', 'gridEnv1_offline_dqn_2022-01-19-16-29-46', 'gridEnv1_offline_dqn_2022-01-19-16-37-38', 'gridEnv1_offline_dqn_2022-01-19-16-45-32', 'gridEnv1_offline_dqn_2022-01-19-16-53-46', 'gridEnv1_offline_dqn_2022-01-19-17-01-37', 'gridEnv1_offline_dqn_2022-01-19-17-09-23', 'gridEnv1_offline_dqn_2022-01-19-17-17-07', 'gridEnv1_offline_dqn_2022-01-19-17-25-02', 'gridEnv1_offline_dqn_2022-01-22-01-23-59', 'gridEnv1_offline_dqn_2022-01-22-01-31-25', 'gridEnv1_offline_dqn_2022-01-22-01-38-16', 'gridEnv1_offline_dqn_2022-01-22-01-45-11', 'gridEnv1_offline_dqn_2022-01-22-01-52-06', 'gridEnv1_offline_dqn_2022-01-22-01-59-00', 'gridEnv1_offline_dqn_2022-01-22-02-06-04', 'gridEnv1_offline_dqn_2022-01-22-02-12-57', 'gridEnv1_offline_dqn_2022-01-22-02-19-53', 'gridEnv1_offline_dqn_2022-01-22-02-26-58', 'gridEnv1_offline_dqn_2022-01-22-02-33-51',
# Boltzmann.
'gridEnv1_offline_dqn_2022-01-19-19-30-29', 'gridEnv1_offline_dqn_2022-01-19-19-38-39', 'gridEnv1_offline_dqn_2022-01-19-19-46-46', 'gridEnv1_offline_dqn_2022-01-19-19-54-45', 'gridEnv1_offline_dqn_2022-01-19-20-02-42', 'gridEnv1_offline_dqn_2022-01-19-20-10-44', 'gridEnv1_offline_dqn_2022-01-19-20-18-30', 'gridEnv1_offline_dqn_2022-01-19-20-26-22', 'gridEnv1_offline_dqn_2022-01-19-20-34-09', 'gridEnv1_offline_dqn_2022-01-19-20-41-58', 'gridEnv1_offline_dqn_2022-01-19-20-49-48', 'gridEnv1_offline_dqn_2022-01-19-20-57-38', 'gridEnv1_offline_dqn_2022-01-19-21-05-27', 'gridEnv1_offline_dqn_2022-01-19-21-13-18', 'gridEnv1_offline_dqn_2022-01-19-21-21-02', 'gridEnv1_offline_dqn_2022-01-19-21-28-52', 'gridEnv1_offline_dqn_2022-01-19-21-36-39', 'gridEnv1_offline_dqn_2022-01-19-21-44-29', 'gridEnv1_offline_dqn_2022-01-19-21-52-20', 'gridEnv1_offline_dqn_2022-01-19-22-00-09', 'gridEnv1_offline_dqn_2022-01-19-22-07-55', 'gridEnv1_offline_dqn_2022-01-22-02-40-47', 'gridEnv1_offline_dqn_2022-01-22-02-47-51', 'gridEnv1_offline_dqn_2022-01-22-02-54-49', 'gridEnv1_offline_dqn_2022-01-22-03-01-51', 'gridEnv1_offline_dqn_2022-01-22-03-08-55', 'gridEnv1_offline_dqn_2022-01-22-03-15-48', 'gridEnv1_offline_dqn_2022-01-22-03-22-55', 'gridEnv1_offline_dqn_2022-01-22-03-30-01', 'gridEnv1_offline_dqn_2022-01-22-03-36-58', 'gridEnv1_offline_dqn_2022-01-22-03-43-59', 'gridEnv1_offline_dqn_2022-01-22-03-51-00', 'gridEnv1_offline_dqn_2022-01-22-03-58-02', 'gridEnv1_offline_dqn_2022-01-22-04-05-01', 'gridEnv1_offline_dqn_2022-01-22-04-12-00', 'gridEnv1_offline_dqn_2022-01-22-04-19-00', 'gridEnv1_offline_dqn_2022-01-22-04-26-06', 'gridEnv1_offline_dqn_2022-01-22-04-33-00', 'gridEnv1_offline_dqn_2022-01-22-04-40-03', 'gridEnv1_offline_dqn_2022-01-22-04-47-04', 'gridEnv1_offline_dqn_2022-01-22-04-54-07', 'gridEnv1_offline_dqn_2022-01-22-05-01-02',

# CQL.
# Dirichlet.
'gridEnv1_offline_cql_2022-01-20-14-25-26', 'gridEnv1_offline_cql_2022-01-20-14-36-46', 'gridEnv1_offline_cql_2022-01-20-14-46-45', 'gridEnv1_offline_cql_2022-01-20-14-56-38', 'gridEnv1_offline_cql_2022-01-20-15-06-26', 'gridEnv1_offline_cql_2022-01-20-15-16-04', 'gridEnv1_offline_cql_2022-01-20-15-25-52', 'gridEnv1_offline_cql_2022-01-22-05-07-54', 'gridEnv1_offline_cql_2022-01-22-05-17-36', 'gridEnv1_offline_cql_2022-01-22-05-27-16', 'gridEnv1_offline_cql_2022-01-22-05-37-01', 'gridEnv1_offline_cql_2022-01-22-05-46-48', 'gridEnv1_offline_cql_2022-01-22-05-56-35', 'gridEnv1_offline_cql_2022-01-22-06-06-15',
# Eps-greedy.
'gridEnv1_offline_cql_2022-01-20-00-08-34', 'gridEnv1_offline_cql_2022-01-20-00-27-32', 'gridEnv1_offline_cql_2022-01-20-00-46-13', 'gridEnv1_offline_cql_2022-01-20-01-05-15', 'gridEnv1_offline_cql_2022-01-20-01-24-35', 'gridEnv1_offline_cql_2022-01-20-01-43-50', 'gridEnv1_offline_cql_2022-01-20-02-02-46', 'gridEnv1_offline_cql_2022-01-20-02-21-32', 'gridEnv1_offline_cql_2022-01-20-02-40-40', 'gridEnv1_offline_cql_2022-01-20-02-59-50', 'gridEnv1_offline_cql_2022-01-20-03-18-40', 'gridEnv1_offline_cql_2022-01-22-06-15-59', 'gridEnv1_offline_cql_2022-01-22-06-25-42', 'gridEnv1_offline_cql_2022-01-22-06-35-28', 'gridEnv1_offline_cql_2022-01-22-06-45-11', 'gridEnv1_offline_cql_2022-01-22-06-54-51', 'gridEnv1_offline_cql_2022-01-22-07-04-33', 'gridEnv1_offline_cql_2022-01-22-07-14-20', 'gridEnv1_offline_cql_2022-01-22-07-24-09', 'gridEnv1_offline_cql_2022-01-22-07-33-50', 'gridEnv1_offline_cql_2022-01-22-07-43-31', 'gridEnv1_offline_cql_2022-01-22-07-53-26',
# Boltzmann.
'gridEnv1_offline_cql_2022-01-20-01-05-19', 'gridEnv1_offline_cql_2022-01-20-01-16-29', 'gridEnv1_offline_cql_2022-01-20-01-27-04', 'gridEnv1_offline_cql_2022-01-20-01-37-35', 'gridEnv1_offline_cql_2022-01-20-01-48-07', 'gridEnv1_offline_cql_2022-01-20-01-58-47', 'gridEnv1_offline_cql_2022-01-20-02-09-24', 'gridEnv1_offline_cql_2022-01-20-02-19-53', 'gridEnv1_offline_cql_2022-01-20-02-30-42', 'gridEnv1_offline_cql_2022-01-20-02-41-09', 'gridEnv1_offline_cql_2022-01-20-02-51-43', 'gridEnv1_offline_cql_2022-01-20-03-02-16', 'gridEnv1_offline_cql_2022-01-20-03-12-53', 'gridEnv1_offline_cql_2022-01-20-03-23-23', 'gridEnv1_offline_cql_2022-01-20-03-33-58', 'gridEnv1_offline_cql_2022-01-20-03-44-31', 'gridEnv1_offline_cql_2022-01-20-03-55-04', 'gridEnv1_offline_cql_2022-01-20-04-05-37', 'gridEnv1_offline_cql_2022-01-20-04-16-14', 'gridEnv1_offline_cql_2022-01-20-04-26-44', 'gridEnv1_offline_cql_2022-01-20-04-37-14', 'gridEnv1_offline_cql_2022-01-22-08-03-07', 'gridEnv1_offline_cql_2022-01-22-08-12-47', 'gridEnv1_offline_cql_2022-01-22-08-22-23', 'gridEnv1_offline_cql_2022-01-22-08-32-07', 'gridEnv1_offline_cql_2022-01-22-08-41-54', 'gridEnv1_offline_cql_2022-01-22-08-51-46', 'gridEnv1_offline_cql_2022-01-22-09-01-32', 'gridEnv1_offline_cql_2022-01-22-09-11-17', 'gridEnv1_offline_cql_2022-01-22-09-20-58', 'gridEnv1_offline_cql_2022-01-22-09-30-37', 'gridEnv1_offline_cql_2022-01-22-09-40-12', 'gridEnv1_offline_cql_2022-01-22-09-49-49', 'gridEnv1_offline_cql_2022-01-22-09-59-30', 'gridEnv1_offline_cql_2022-01-22-10-09-15', 'gridEnv1_offline_cql_2022-01-22-10-18-59', 'gridEnv1_offline_cql_2022-01-22-10-28-43', 'gridEnv1_offline_cql_2022-01-22-10-38-19', 'gridEnv1_offline_cql_2022-01-22-10-48-12', 'gridEnv1_offline_cql_2022-01-22-10-57-53', 'gridEnv1_offline_cql_2022-01-22-11-07-38', 'gridEnv1_offline_cql_2022-01-22-11-17-20',

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
