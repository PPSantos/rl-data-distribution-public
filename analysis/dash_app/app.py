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

ENVS = ['GridEnv1'] #, 'GridEnv2', 'MultiPath']
ALGORITHMS = ['DQN'] #, 'CQL']
DATASET_TYPES = ['Dirichlet', 'Eps-greedy', 'Boltzmann']

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

        #'CQL': {},

    },
    # 'GridEnv2': {},
    # 'MultiPath': {},
}

def load_data(exp_ids):
    print('Pre-loading data.')

    nested_dict = lambda: collections.defaultdict(nested_dict)
    metrics_dict = nested_dict()
    dataset_metrics_dict = nested_dict()
    for env_id, env_data in exp_ids.items():
        print('env_id=', env_id)
        for algo_id, algo_data in env_data.items():
            print('algo_id=', algo_id)
            for dataset_type_id, dataset_type_data in algo_data.items():
                print('dataset_type_id=', dataset_type_id)

                metrics_list = []
                dataset_metrics_list = []
                for exp_id in dataset_type_data:

                    # Load algorithm metrics.
                    exp_metrics_path = PLOTS_FOLDER_PATH + exp_id + '/scalar_metrics.json'
                    with open(exp_metrics_path, 'r') as f:
                        d = json.load(f)
                    f.close()
                    metrics_list.append(d)

                    # Load dataset metrics.
                    exp_folder_path = DATA_FOLDER_PATH + exp_id + '.tar.gz'
                    tar = tarfile.open(exp_folder_path)
                    data_file = tar.extractfile("{0}/dataset_info.json".format(exp_id))
                    dataset_info = json.load(data_file)
                    dataset_info = json.loads(dataset_info)
                    dataset_metrics_list.append(dataset_info)

                metrics_dict[env_id][algo_id][dataset_type_id] = metrics_list
                dataset_metrics_dict[env_id][algo_id][dataset_type_id] = dataset_metrics_list

    print('Finished pre-loading data.')

    return metrics_dict, dataset_metrics_dict

ALGO_METRICS, DATASET_METRICS = load_data(EXP_IDS)

print(ALGO_METRICS['GridEnv1']['DQN']['Dirichlet'])

app = dash.Dash(__name__,
                meta_tags=[
                    {
                        'name': 'rl-data-distribution',
                        'content': 'Understanding the impact of data distribution on Q-learning with function approximation.'
                    },
                    {
                        'http-equiv': 'X-UA-Compatible',
                        'content': 'IE=edge'
                    },
                    {
                    'name': 'viewport',
                    'content': 'width=device-width, initial-scale=1.0'
                    }]
                )

app.title = 'RL data distribution'
server = app.server
COLORS = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']


def div_graph(name):

    components = []
    components.append(html.Div(id="div-{0}-graph".format(name), className="twelve columns"))

    return html.Div(
        className="row",
        children=components,
    )


page_layout = []

# Top row.
top_row_layout = html.Div(
            [
                html.H2(
                    "RL Data Distribution",
                    id="title",
                    className="eight columns",
                    style={"margin-left": "3%"},
                ),
            ],
            className="banner row",
        )
page_layout.append(top_row_layout)

runs_picker_layout_json = html.Div(
            className="container",
            style={"padding": "35px 25px"},
            children=[
                html.Div(
                    className="row",
                    style={"margin": "8px 0px"},
                    children=[
                        html.Div(
                            className="four columns",
                            children=[
                                    html.P("Environment:"),
                                    dcc.Dropdown(
                                        id='env-picker',
                                        options=[{'label': env, 'value': env} for env in ENVS],
                                        value='GridEnv1',
                                        multi=False,
                                        style={'color': '#3b505e'},
                                        className="dropdown-box-third",
                                    )
                            ]
                        ),
                        html.Div(
                            className="four columns",
                            children=[
                                    html.P("Algorithm:"),
                                    dcc.Dropdown(
                                        id='algorithm-picker',
                                        options=[{'label': algo, 'value': algo} for algo in ALGORITHMS],
                                        value=['DQN'],
                                        multi=True,
                                        style={'color': '#3b505e'},
                                        className="dropdown-box-third",
                                    )
                            ]
                        ),
                        html.Div(
                            className="four columns",
                            children=[
                                    html.P("Dataset type:"),
                                    dcc.Dropdown(
                                        id='dataset-picker',
                                        options=[{'label': d, 'value': d} for d in DATASET_TYPES],
                                        value=['Dirichlet'],
                                        multi=True,
                                        style={'color': '#3b505e'},
                                        className="dropdown-box-third",
                                    )
                            ]
                        ),
                    ]),
            ])
page_layout.append(runs_picker_layout_json)

# Plots.
page_layout.append(html.Div(className="container", children=[div_graph('entropy')]))


app.layout = html.Div(
    style={"height": "100%"},
    children=page_layout,
)

@app.callback(
    Output("div-entropy-graph", "children"),
    [
        Input("env-picker", "value"),
        Input("algorithm-picker", "value"),
        Input("dataset-picker", "value"),
    ],
)
def callback_entropy_plot(env, algorithms, dataset_types):

    print('env', env)
    print('algorithms', algorithms)
    print('dataset_types', dataset_types)

    layout = go.Layout(
        #title='Entropy',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Entropy'},
        yaxis={"title": 'Q-values error'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    traces = []

    for (c_idx, algo) in enumerate(sorted(algorithms)):

        algo_data_X = []
        algo_data_Y = []
        hover_text = []

        for dataset_type in dataset_types:
            algo_data_Y.extend([y['qvals_avg_error'] for y in ALGO_METRICS[env][algo][dataset_type]])
            algo_data_X.extend([x['dataset_entropy'] for x in DATASET_METRICS[env][algo][dataset_type]])
            h_texts = [f"Entropy: {x['dataset_entropy']:.2f}<br>Q-value error: {y['qvals_avg_error']:.2f}"
                                        for x, y in zip(DATASET_METRICS[env][algo][dataset_type],
                                        ALGO_METRICS[env][algo][dataset_type])]
            hover_text.extend(h_texts)

        traces.append(go.Scatter(
            x=algo_data_X,
            y=algo_data_Y,
            showlegend=True,
            marker_color=COLORS[c_idx],
            mode='markers',
            name=algo,
            hovertext=hover_text,
            hoverinfo="text",
        ))

    figure = go.Figure(data=traces, layout=layout)
    figure.update_layout(yaxis_type="log")

    return dcc.Graph(figure=figure, id='entropy-graph')

if __name__ == "__main__":
    app.run_server(debug=True)
