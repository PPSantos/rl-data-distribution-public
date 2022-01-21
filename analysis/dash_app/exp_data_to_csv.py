import os
import json
import pathlib
import tarfile
import collections

import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import numpy as np
import pandas as pd

# Path to folder containing data files.
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/plots/'


EXP_IDS = {
    'GridEnv1': {
        'DQN': {
            'Dirichlet': [
                'gridEnv1_offline_dqn_2022-01-18-22-07-14', 'gridEnv1_offline_dqn_2022-01-18-22-15-26', 'gridEnv1_offline_dqn_2022-01-18-22-23-08', 'gridEnv1_offline_dqn_2022-01-18-22-31-01', 'gridEnv1_offline_dqn_2022-01-18-22-38-56', 'gridEnv1_offline_dqn_2022-01-18-22-46-39', 'gridEnv1_offline_dqn_2022-01-18-22-54-30',
                'gridEnv1_offline_dqn_2022-01-18-23-02-12', 'gridEnv1_offline_dqn_2022-01-18-23-09-59', 'gridEnv1_offline_dqn_2022-01-18-23-17-44', 'gridEnv1_offline_dqn_2022-01-18-23-25-34', 'gridEnv1_offline_dqn_2022-01-18-23-33-21', 'gridEnv1_offline_dqn_2022-01-18-23-41-07', 'gridEnv1_offline_dqn_2022-01-18-23-49-02',
            ],
            'Eps-greedy': [
                'gridEnv1_offline_dqn_2022-01-19-16-05-47', 'gridEnv1_offline_dqn_2022-01-19-16-14-01', 'gridEnv1_offline_dqn_2022-01-19-16-21-57', 'gridEnv1_offline_dqn_2022-01-19-16-29-46', 'gridEnv1_offline_dqn_2022-01-19-16-37-38', 'gridEnv1_offline_dqn_2022-01-19-16-45-32', 'gridEnv1_offline_dqn_2022-01-19-16-53-46', 'gridEnv1_offline_dqn_2022-01-19-17-01-37', 'gridEnv1_offline_dqn_2022-01-19-17-09-23', 'gridEnv1_offline_dqn_2022-01-19-17-17-07', 'gridEnv1_offline_dqn_2022-01-19-17-25-02', 'gridEnv1_offline_dqn_2022-01-19-17-32-51', 'gridEnv1_offline_dqn_2022-01-19-17-40-41', 'gridEnv1_offline_dqn_2022-01-19-17-48-27', 'gridEnv1_offline_dqn_2022-01-19-17-56-19', 'gridEnv1_offline_dqn_2022-01-19-18-04-20', 'gridEnv1_offline_dqn_2022-01-19-18-12-08', 'gridEnv1_offline_dqn_2022-01-19-18-19-59', 'gridEnv1_offline_dqn_2022-01-19-18-27-43', 'gridEnv1_offline_dqn_2022-01-19-18-35-27', 'gridEnv1_offline_dqn_2022-01-19-18-43-09', 'gridEnv1_offline_dqn_2022-01-19-18-50-57',
            ],
            'Boltzmann': [
                'gridEnv1_offline_dqn_2022-01-19-19-30-29', 'gridEnv1_offline_dqn_2022-01-19-19-38-39', 'gridEnv1_offline_dqn_2022-01-19-19-46-46', 'gridEnv1_offline_dqn_2022-01-19-19-54-45', 'gridEnv1_offline_dqn_2022-01-19-20-02-42', 'gridEnv1_offline_dqn_2022-01-19-20-10-44', 'gridEnv1_offline_dqn_2022-01-19-20-18-30', 'gridEnv1_offline_dqn_2022-01-19-20-26-22', 'gridEnv1_offline_dqn_2022-01-19-20-34-09', 'gridEnv1_offline_dqn_2022-01-19-20-41-58', 'gridEnv1_offline_dqn_2022-01-19-20-49-48', 'gridEnv1_offline_dqn_2022-01-19-20-57-38', 'gridEnv1_offline_dqn_2022-01-19-21-05-27', 'gridEnv1_offline_dqn_2022-01-19-21-13-18', 'gridEnv1_offline_dqn_2022-01-19-21-21-02', 'gridEnv1_offline_dqn_2022-01-19-21-28-52', 'gridEnv1_offline_dqn_2022-01-19-21-36-39', 'gridEnv1_offline_dqn_2022-01-19-21-44-29', 'gridEnv1_offline_dqn_2022-01-19-21-52-20', 'gridEnv1_offline_dqn_2022-01-19-22-00-09', 'gridEnv1_offline_dqn_2022-01-19-22-07-55', 'gridEnv1_offline_dqn_2022-01-19-22-15-48', 'gridEnv1_offline_dqn_2022-01-19-22-23-38', 'gridEnv1_offline_dqn_2022-01-19-22-31-28', 'gridEnv1_offline_dqn_2022-01-19-22-39-19', 'gridEnv1_offline_dqn_2022-01-19-22-47-12', 'gridEnv1_offline_dqn_2022-01-19-22-55-03', 'gridEnv1_offline_dqn_2022-01-19-23-02-55', 'gridEnv1_offline_dqn_2022-01-19-23-10-43', 'gridEnv1_offline_dqn_2022-01-19-23-18-39', 'gridEnv1_offline_dqn_2022-01-19-23-26-31', 'gridEnv1_offline_dqn_2022-01-19-23-34-18', 'gridEnv1_offline_dqn_2022-01-19-23-42-10', 'gridEnv1_offline_dqn_2022-01-19-23-50-02', 'gridEnv1_offline_dqn_2022-01-19-23-57-47', 'gridEnv1_offline_dqn_2022-01-20-00-05-37', 'gridEnv1_offline_dqn_2022-01-20-00-13-21', 'gridEnv1_offline_dqn_2022-01-20-00-21-09', 'gridEnv1_offline_dqn_2022-01-20-00-29-01', 'gridEnv1_offline_dqn_2022-01-20-00-36-48', 'gridEnv1_offline_dqn_2022-01-20-00-44-37', 'gridEnv1_offline_dqn_2022-01-20-00-52-30',
            ],
        },

        'CQL': {
            'Dirichlet': [
                'gridEnv1_offline_cql_2022-01-20-14-25-26', 'gridEnv1_offline_cql_2022-01-20-14-36-46', 'gridEnv1_offline_cql_2022-01-20-14-46-45', 'gridEnv1_offline_cql_2022-01-20-14-56-38', 'gridEnv1_offline_cql_2022-01-20-15-06-26', 'gridEnv1_offline_cql_2022-01-20-15-16-04', 'gridEnv1_offline_cql_2022-01-20-15-25-52', 'gridEnv1_offline_cql_2022-01-20-15-35-44', 'gridEnv1_offline_cql_2022-01-20-15-45-38', 'gridEnv1_offline_cql_2022-01-20-15-55-21', 'gridEnv1_offline_cql_2022-01-20-16-04-59', 'gridEnv1_offline_cql_2022-01-20-16-14-38', 'gridEnv1_offline_cql_2022-01-20-16-24-20', 'gridEnv1_offline_cql_2022-01-20-16-34-06'
            ],
            'Eps-greedy': [
                'gridEnv1_offline_cql_2022-01-20-14-25-26', # FIXME
            ],
            'Boltzmann': [
                'gridEnv1_offline_cql_2022-01-20-01-05-19', 'gridEnv1_offline_cql_2022-01-20-01-16-29', 'gridEnv1_offline_cql_2022-01-20-01-27-04', 'gridEnv1_offline_cql_2022-01-20-01-37-35', 'gridEnv1_offline_cql_2022-01-20-01-48-07', 'gridEnv1_offline_cql_2022-01-20-01-58-47', 'gridEnv1_offline_cql_2022-01-20-02-09-24', 'gridEnv1_offline_cql_2022-01-20-02-19-53', 'gridEnv1_offline_cql_2022-01-20-02-30-42', 'gridEnv1_offline_cql_2022-01-20-02-41-09', 'gridEnv1_offline_cql_2022-01-20-02-51-43', 'gridEnv1_offline_cql_2022-01-20-03-02-16', 'gridEnv1_offline_cql_2022-01-20-03-12-53', 'gridEnv1_offline_cql_2022-01-20-03-23-23', 'gridEnv1_offline_cql_2022-01-20-03-33-58', 'gridEnv1_offline_cql_2022-01-20-03-44-31', 'gridEnv1_offline_cql_2022-01-20-03-55-04', 'gridEnv1_offline_cql_2022-01-20-04-05-37', 'gridEnv1_offline_cql_2022-01-20-04-16-14', 'gridEnv1_offline_cql_2022-01-20-04-26-44', 'gridEnv1_offline_cql_2022-01-20-04-37-14', 'gridEnv1_offline_cql_2022-01-20-04-47-41', 'gridEnv1_offline_cql_2022-01-20-04-58-12', 'gridEnv1_offline_cql_2022-01-20-05-08-45', 'gridEnv1_offline_cql_2022-01-20-05-19-18', 'gridEnv1_offline_cql_2022-01-20-05-29-53', 'gridEnv1_offline_cql_2022-01-20-05-40-30', 'gridEnv1_offline_cql_2022-01-20-05-51-06', 'gridEnv1_offline_cql_2022-01-20-06-01-39', 'gridEnv1_offline_cql_2022-01-20-06-12-08', 'gridEnv1_offline_cql_2022-01-20-06-22-47', 'gridEnv1_offline_cql_2022-01-20-06-33-23', 'gridEnv1_offline_cql_2022-01-20-06-43-54', 'gridEnv1_offline_cql_2022-01-20-06-54-26', 'gridEnv1_offline_cql_2022-01-20-07-04-58', 'gridEnv1_offline_cql_2022-01-20-07-15-29', 'gridEnv1_offline_cql_2022-01-20-07-25-58', 'gridEnv1_offline_cql_2022-01-20-07-36-33', 'gridEnv1_offline_cql_2022-01-20-07-47-15', 'gridEnv1_offline_cql_2022-01-20-07-57-48', 'gridEnv1_offline_cql_2022-01-20-08-08-44', 'gridEnv1_offline_cql_2022-01-20-08-19-14'
            ],
        },

    },
    # 'GridEnv2': {},
    # 'MultiPath': {},
}

def data_to_csv(exp_ids):
    print('Pre-loading data.')

    df_rows = []

    for env_id, env_data in exp_ids.items():
        print('env_id:', env_id)
        for algo_id, algo_data in env_data.items():
            print('algo_id:', algo_id)
            for dataset_type_id, dataset_type_data in algo_data.items():
                print('dataset_type_id:', dataset_type_id)

                for exp_id in dataset_type_data:

                    row_data = {}
                    row_data['id'] = exp_id
                    row_data['env_id'] = env_id
                    row_data['algo_id'] = algo_id
                    row_data['dataset_type_id'] = dataset_type_id

                    # Load algorithm metrics.
                    exp_metrics_path = PLOTS_FOLDER_PATH + exp_id + '/scalar_metrics.json'
                    with open(exp_metrics_path, 'r') as f:
                        d = json.load(f)
                    f.close()

                    row_data['qvals_avg_error'] = d['qvals_avg_error']
                    row_data['qvals_summed_error'] = d['qvals_summed_error']
                    row_data['rollouts_rewards_final'] = d['rollouts_rewards_final']

                    # Load dataset metrics.
                    exp_folder_path = DATA_FOLDER_PATH + exp_id + '.tar.gz'
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

                    info_text = f"Exp-id: {row_data['id']}<br>Environment: {row_data['env_id']}<br>Algorithm: {row_data['algo_id']}<br>Dataset type: {row_data['dataset_type_id']}<br>Dataset coverage: {row_data['dataset_coverage']:.2f}<br>Q-values avg error: {row_data['qvals_avg_error']:.2f}<br>Q-values summed error: {row_data['qvals_summed_error']:.2f}<br>Rollouts rewards: {row_data['rollouts_rewards_final']:.2f}"
                    row_data['info_text'] = info_text

                    df_rows.append(row_data)

    print('Finished parsing data.')

    df = pd.DataFrame(df_rows)
    print(df['dataset_coverage'])

    df.to_csv(DATA_FOLDER_PATH + 'parsed_data.csv')

if __name__ == "__main__":
    data_to_csv(EXP_IDS)