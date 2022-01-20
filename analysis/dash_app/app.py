import os
import json
import pathlib
import tarfile

import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import numpy as np
import pandas as pd

# Path to folder containing data files.
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'
PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/analysis/plots/'

ENVS = ['GridEnv1', 'GridEnv2', 'MultiPath']
ALGORITHMS = ['DQN', 'CQL']
DATASET_TYPES = ['Dirichlet', 'Eps-greedy', 'Boltzmann']

EXP_IDS = {
    'gridEnv1': {
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

        'CQL': {},

    },
    'gridEnv2': {},
    'MultiPath': {},
}

def load_data(exp_ids):

    algo_metrics_dict = {}
    dataset_metrics_dict = {}
    for env_id, env_data in exp_ids.items():
        for algo_id, algo_data in env_data.items():
            for dataset_type_id, dataset_type_data in algo_data.items():

                algo_metrics_list = []
                dataset_metrics_list = []
                for exp_id in dataset_type_data:
                    exp_metrics_path = PLOTS_FOLDER_PATH + exp_id + '/scalar_metrics.json'
                    with open(exp_metrics_path, 'r') as f:
                        d = json.load(f)
                    f.close()
                    algo_metrics_list.append(d)

                    exp_folder_path = DATA_FOLDER_PATH + exp_id + '.tar.gz'

                    tar = tarfile.open(exp_folder_path)
                    data_file = tar.extractfile("{0}/dataset_info.json".format(exp_id))

                    dataset_info = json.load(data_file)
                    dataset_info = json.loads(dataset_info)

                    dataset_metrics_list.append(dataset_info)

                algo_metrics_dict[env_id][algo_id][dataset_type_id] = algo_metrics_list
                dataset_metrics_dict[env_id][algo_id][dataset_type_id] = dataset_metrics_list

    return algo_metrics_dict, dataset_metrics_dict
ALGO_METRICS, DATASET_METRICS = load_data(EXP_IDS)

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
                                        value='DQN',
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
                                        value='dirichlet',
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
def callback_rl_actions(env, algorithms, dataset_types):

    print('env', env)
    print('algorithms', algorithms)
    print('dataset_types', dataset_types)

    layout = go.Layout(
        title='Entropy',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Entropy'},
        yaxis={"title": 'Q-values error'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    env_exp_ids = EXP_IDS[env]

    traces = []

    for algo in algorithms:

        for dataset_type in dataset_types:

            data = []
            for exp_id in env_exp_ids[algo][dataset_type]:
                pass

            traces.append(go.Scatter(
                x=X,
                y=Y,
                #mode="lines",
                name=os.path.basename(file_name),
                showlegend=True,
            ))

    traces = []
    for file_name in sorted(runs):

        d = data[file_name]

        if d:
            
            Y = d['rl_actions']
            X = np.linspace(1, len(Y), len(Y))

            Y = np.array(Y)
            Y = Y.flatten()

            traces.append(go.Scatter(
                x=X,
                y=Y,
                #mode="lines",
                name=os.path.basename(file_name),
                showlegend=True,
            ))

    figure = go.Figure(data=traces, layout=layout)

    return dcc.Graph(figure=figure, id='rl_actions-graph')

"""
def update_graph(
    graph_id,
    var_name,
    graph_title,
    xaxis_title,
    yaxis_title,
    data,
    runs
):

    layout = go.Layout(
        title=graph_title,
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": xaxis_title},
        yaxis={"title": yaxis_title},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    traces = []
    for file_name in sorted(runs):

        d = data[file_name] 

        if d:
            
            Y = d[var_name]
            X = np.linspace(1, len(Y), len(Y))

            traces.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                name=os.path.basename(file_name),
                showlegend=True,
            ))

    figure = go.Figure(data=traces, layout=layout)

    return dcc.Graph(figure=figure, id=graph_id)


@app.callback(
    Output("div-rewards-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("runs-picker-json", "value")
    ],
)
def callback_rewards(data, runs):
        
    graph = update_graph(
        graph_id="rewards-graph",
        var_name='rewards',
        graph_title='Rewards',
        xaxis_title='Cycle',
        yaxis_title='Reward',
        data=data,
        runs=runs,
    )

    return [graph]


@app.callback(
    Output("div-vehicles-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("runs-picker-json", "value")
    ],
)
def callback_vehicles(data, runs):

    graph = update_graph(
        graph_id="vehicles-graph",
        var_name='vehicles',
        graph_title='Avg. number of vehicles',
        xaxis_title='Cycle',
        yaxis_title='Avg. number of vehicles',
        data=data,
        runs=runs,
    )

    return [graph]


@app.callback(
    Output("div-velocities-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("runs-picker-json", "value")
    ],
)
def callback_velocities(data, runs):

    graph = update_graph(
        graph_id="velocities-graph",
        var_name='velocities',
        graph_title='Avg. velocities',
        xaxis_title='Cycle',
        yaxis_title='Avg. velocity',
        data=data,
        runs=runs,
    )

    return [graph]


@app.callback(
    Output("div-observation_space-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("runs-picker-json", "value"),
        Input("time-selector", "value"),
    ],
)
def callback_observation_space(data, runs, time_range):

    low_bound = time_range[0]
    high_bound = time_range[1]

    layout = go.Layout(
        title='Observation space (cycle range = {0})'.format(time_range),
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Speed'},
        yaxis={"title": 'Count'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    # Only works for one run.
    if not runs or len(runs) > 1:
        figure = go.Figure(data=[], layout=layout)
        return dcc.Graph(figure=figure, id='observation_space-graph')

    traces = []

    file_name = runs[0]

    d = data[file_name]

    if d and 'observation_spaces' in d.keys():
        
        obs = d['observation_spaces']

        phase_0_X = []
        phase_0_Y = []
        phase_1_X = []
        phase_1_Y = []

        for t, observation_space in enumerate(obs):

            # Only one TSC.
            observation_space = observation_space[0]

            phase_0_X.append(observation_space[0][0]) # speed phase 0
            phase_0_Y.append(observation_space[0][1]) # count phase 0
            phase_1_X.append(observation_space[1][0]) # speed phase 1
            phase_1_Y.append(observation_space[1][1]) # count phase 1

        traces.append(go.Scatter(
            x=phase_0_X[low_bound:high_bound],
            y=phase_0_Y[low_bound:high_bound],
            mode="markers",
            name="phase 0",
            showlegend=True,
        ))

        traces.append(go.Scatter(
            x=phase_1_X[low_bound:high_bound],
            y=phase_1_Y[low_bound:high_bound],
            mode="markers",
            name="phase 1",
            showlegend=True,
        ))

    figure = go.Figure(data=traces, layout=layout)

    return dcc.Graph(figure=figure, id='observation_space-graph')


@app.callback(
    Output("div-explored-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("runs-picker-json", "value")
    ],
)
def callback_explored(data, runs):

    layout = go.Layout(
        title='Explored',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Cycle'},
        yaxis={"title": 'Explored?'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    traces = []
    for file_name in sorted(runs):

        d = data[file_name]

        if d:
            
            Y = d['explored']
            X = np.linspace(1, len(Y), len(Y))

            Y = np.array(Y)
            Y = np.where(Y==True, 1, Y) 
            Y = np.where(Y==False, 0, Y) 

            traces.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                #marker=dict(size=5),
                name=os.path.basename(file_name),
                showlegend=True,
            ))

    figure = go.Figure(data=traces, layout=layout)

    return dcc.Graph(figure=figure, id='explored-graph')

@app.callback(
    Output("div-Q_distances-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("runs-picker-json", "value")
    ],
)
def callback_Q_distances(data, runs):

    layout = go.Layout(
        title='Q-tables distances',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Cycle'},
        yaxis={"title": 'Abs sum distance'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    traces = []
    for file_name in sorted(runs):

        d = data[file_name]

        if d and 'Q_distances' in d.keys():
            
            Y = d['Q_distances']
            X = np.linspace(1, len(Y), len(Y))

            traces.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                name=os.path.basename(file_name),
                showlegend=True,
            ))

    figure = go.Figure(data=traces, layout=layout)

    return dcc.Graph(figure=figure, id='Q_distances-graph')


@app.callback(
    Output("div-visited_states-graph", "children"),
    [
        Input("json-data-storage", "data"),
        Input("env-picker", "value")
    ],
)
def callback_visited_states(data, runs):

    layout = go.Layout(
        title='(Newly) visited states',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Cycle'},
        yaxis={"title": 'Visited a new state?'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    traces = []
    for file_name in sorted(runs):

        d = data[file_name]

        if d and 'visited_states' in d.keys():
            
            Y = d['visited_states']

            Y = [1 if i is not None else 0 for i in Y]
            X = np.linspace(1, len(Y), len(Y))

            traces.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                name=os.path.basename(file_name),
                showlegend=True,
            ))

    figure = go.Figure(data=traces, layout=layout)

    return dcc.Graph(figure=figure, id='visited_states-graph')


@app.callback(
    Output("runs-picker-json", "options"),
    [
        Input("runs-picker-json", "value"),
        Input("json-filenames-storage", "data")
    ]
)
def disable_dropdown_options_json(selected_files, filenames):

    if filenames:
        # Limit up to 5.
        if len(selected_files) >= 5:
            opts = [{'label': os.path.basename(f), 'value': f, 'disabled': True}
            for f in sorted(filenames)]
        else:
            opts = [{'label': os.path.basename(f), 'value': f}
            for f in sorted(filenames)]

        return opts

    else:
        return []


@app.callback(
    Output("json-data-update-interval", "interval"),
    [Input("json-live-update-selector", "on")],
)
def update_interval_log_update(val):

    if val:
        return 5 * 1000 # (5 seconds)
    else:
        return 24 * 60 * 60 * 1000 # (24 hours)


@app.callback(
    Output("csv-filenames-storage", "data"),
    [   
        Input("csv-filenames-update-interval", "n_intervals"),
    ]
)
def update_filenames_csv(_):

    csv_files = []
    for path in Path(PATH_FOLDER).rglob('*.csv'):
        csv_files.append(str(path))

    return csv_files


@app.callback(
    Output("runs-picker-csv", "options"),
    [
        Input("runs-picker-csv", "value"),
        Input("csv-filenames-storage", "data")
    ]
)
def disable_dropdown_options_csv(selected_files, filenames):

    if filenames:
        # Limit up to 5.
        if len(selected_files) >= 5:
            opts = [{'label': os.path.basename(f), 'value': f, 'disabled': True}
            for f in sorted(filenames)]
        else:
            opts = [{'label': os.path.basename(f), 'value': f}
            for f in sorted(filenames)]

        return opts

    else:
        return []


@app.callback(
    [
        Output("div-waiting_time-graph", "children"),
        Output("div-avg_travel_time-graph", "children"),
        Output("div-throughput-graph", "children"),
    ],
    [
        Input("runs-picker-csv", "value")
    ],
)
def callback_csv_graphs(selected_filenames):

    traces_waiting_time = []
    traces_avg_travel_time = []
    traces_throughput = []
    for filename in sorted(selected_filenames):

        if filename:

            df = get_emissions(filename)
            df_2 = get_vehicles(df)
            intervals = np.arange(0, df_2['finish'].max(), CYCLE_TIME)
            df_3 = df_2.groupby(pd.cut(df_2["finish"], intervals)).mean()

            Y = df_3['waiting'].values
            X = np.linspace(1, len(Y), len(Y))

            traces_waiting_time.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                name=os.path.basename(filename),
                showlegend=True,
            ))

            Y = df_3['total'].values
            X = np.linspace(1, len(Y), len(Y))

            traces_avg_travel_time.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                name=os.path.basename(filename),
                showlegend=True,
            ))

            df_throughput = get_throughput(df)

            intervals = np.arange(0, df_throughput['time'].max(), CYCLE_TIME)
            df_4 = df_throughput.groupby(pd.cut(df_throughput["time"], intervals)).count()

            Y = df_4['time'].values
            X = np.linspace(1, len(Y), len(Y))

            traces_throughput.append(go.Scatter(
                x=X,
                y=Y,
                mode="lines",
                name=os.path.basename(filename),
                showlegend=True,
            ))

    graphs = []

    layout = go.Layout(
        title='Waiting time',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Cycle'},
        yaxis={"title": 'Avg. waiting time (s)'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )
    figure = go.Figure(data=traces_waiting_time, layout=layout)
    graphs.append(dcc.Graph(figure=figure, id='waiting_time-graph'))

    layout = go.Layout(
        title='Travel time',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Cycle'},
        yaxis={"title": 'Avg. travel time (s)'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )
    figure = go.Figure(data=traces_avg_travel_time, layout=layout)
    graphs.append(dcc.Graph(figure=figure, id='avg_travel_time-graph'))

    layout = go.Layout(
        title='Throughput',
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": 'Cycle'},
        yaxis={"title": '#cars per cycle'},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )
    figure = go.Figure(data=traces_throughput, layout=layout)
    graphs.append(dcc.Graph(figure=figure, id='throughput-graph'))

    return graphs """


if __name__ == "__main__":
    app.run_server(debug=True)