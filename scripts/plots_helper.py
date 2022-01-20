import pathlib
import tarfile

from analysis.plots import main as plots
from scripts.run import VAL_ITER_DATA

ENV_NAME = 'gridEnv1'
EXP_IDS = [
# Dirichlet: coverage = False
'gridEnv1_offline_dqn_2022-01-18-22-07-14', 'gridEnv1_offline_dqn_2022-01-18-22-15-26', 'gridEnv1_offline_dqn_2022-01-18-22-23-08', 'gridEnv1_offline_dqn_2022-01-18-22-31-01', 'gridEnv1_offline_dqn_2022-01-18-22-38-56', 'gridEnv1_offline_dqn_2022-01-18-22-46-39', 'gridEnv1_offline_dqn_2022-01-18-22-54-30',
# Dirichlet: coverage = True
'gridEnv1_offline_dqn_2022-01-18-23-02-12', 'gridEnv1_offline_dqn_2022-01-18-23-09-59', 'gridEnv1_offline_dqn_2022-01-18-23-17-44', 'gridEnv1_offline_dqn_2022-01-18-23-25-34', 'gridEnv1_offline_dqn_2022-01-18-23-33-21', 'gridEnv1_offline_dqn_2022-01-18-23-41-07', 'gridEnv1_offline_dqn_2022-01-18-23-49-02',
# Eps-greedy.
'gridEnv1_offline_dqn_2022-01-19-16-05-47', 'gridEnv1_offline_dqn_2022-01-19-16-14-01', 'gridEnv1_offline_dqn_2022-01-19-16-21-57', 'gridEnv1_offline_dqn_2022-01-19-16-29-46', 'gridEnv1_offline_dqn_2022-01-19-16-37-38', 'gridEnv1_offline_dqn_2022-01-19-16-45-32', 'gridEnv1_offline_dqn_2022-01-19-16-53-46', 'gridEnv1_offline_dqn_2022-01-19-17-01-37', 'gridEnv1_offline_dqn_2022-01-19-17-09-23', 'gridEnv1_offline_dqn_2022-01-19-17-17-07', 'gridEnv1_offline_dqn_2022-01-19-17-25-02', 'gridEnv1_offline_dqn_2022-01-19-17-32-51', 'gridEnv1_offline_dqn_2022-01-19-17-40-41', 'gridEnv1_offline_dqn_2022-01-19-17-48-27', 'gridEnv1_offline_dqn_2022-01-19-17-56-19', 'gridEnv1_offline_dqn_2022-01-19-18-04-20', 'gridEnv1_offline_dqn_2022-01-19-18-12-08', 'gridEnv1_offline_dqn_2022-01-19-18-19-59', 'gridEnv1_offline_dqn_2022-01-19-18-27-43', 'gridEnv1_offline_dqn_2022-01-19-18-35-27', 'gridEnv1_offline_dqn_2022-01-19-18-43-09', 'gridEnv1_offline_dqn_2022-01-19-18-50-57',
# Boltzmann.
'gridEnv1_offline_dqn_2022-01-19-19-30-29', 'gridEnv1_offline_dqn_2022-01-19-19-38-39', 'gridEnv1_offline_dqn_2022-01-19-19-46-46', 'gridEnv1_offline_dqn_2022-01-19-19-54-45', 'gridEnv1_offline_dqn_2022-01-19-20-02-42', 'gridEnv1_offline_dqn_2022-01-19-20-10-44', 'gridEnv1_offline_dqn_2022-01-19-20-18-30', 'gridEnv1_offline_dqn_2022-01-19-20-26-22', 'gridEnv1_offline_dqn_2022-01-19-20-34-09', 'gridEnv1_offline_dqn_2022-01-19-20-41-58', 'gridEnv1_offline_dqn_2022-01-19-20-49-48', 'gridEnv1_offline_dqn_2022-01-19-20-57-38', 'gridEnv1_offline_dqn_2022-01-19-21-05-27', 'gridEnv1_offline_dqn_2022-01-19-21-13-18', 'gridEnv1_offline_dqn_2022-01-19-21-21-02', 'gridEnv1_offline_dqn_2022-01-19-21-28-52', 'gridEnv1_offline_dqn_2022-01-19-21-36-39', 'gridEnv1_offline_dqn_2022-01-19-21-44-29', 'gridEnv1_offline_dqn_2022-01-19-21-52-20', 'gridEnv1_offline_dqn_2022-01-19-22-00-09', 'gridEnv1_offline_dqn_2022-01-19-22-07-55', 'gridEnv1_offline_dqn_2022-01-19-22-15-48', 'gridEnv1_offline_dqn_2022-01-19-22-23-38', 'gridEnv1_offline_dqn_2022-01-19-22-31-28', 'gridEnv1_offline_dqn_2022-01-19-22-39-19', 'gridEnv1_offline_dqn_2022-01-19-22-47-12', 'gridEnv1_offline_dqn_2022-01-19-22-55-03', 'gridEnv1_offline_dqn_2022-01-19-23-02-55', 'gridEnv1_offline_dqn_2022-01-19-23-10-43', 'gridEnv1_offline_dqn_2022-01-19-23-18-39', 'gridEnv1_offline_dqn_2022-01-19-23-26-31', 'gridEnv1_offline_dqn_2022-01-19-23-34-18', 'gridEnv1_offline_dqn_2022-01-19-23-42-10', 'gridEnv1_offline_dqn_2022-01-19-23-50-02', 'gridEnv1_offline_dqn_2022-01-19-23-57-47', 'gridEnv1_offline_dqn_2022-01-20-00-05-37', 'gridEnv1_offline_dqn_2022-01-20-00-13-21', 'gridEnv1_offline_dqn_2022-01-20-00-21-09', 'gridEnv1_offline_dqn_2022-01-20-00-29-01', 'gridEnv1_offline_dqn_2022-01-20-00-36-48', 'gridEnv1_offline_dqn_2022-01-20-00-44-37', 'gridEnv1_offline_dqn_2022-01-20-00-52-30',
]

if __name__ == "__main__":

    for exp_id in EXP_IDS:
        plots(exp_id=exp_id+'.tar.gz', val_iter_exp=VAL_ITER_DATA[ENV_NAME])
