import os
import pathlib

from PIL import Image

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px

import numpy as np
import pandas as pd

###########################################################################
# Absolute path to folder containing experiments data.
CSV_PATH = '/home/pedrosantos/git/rl-data-distribution/data/parsed_data.csv'
DATA = pd.read_csv(CSV_PATH)
# Absolute path to folder containing plots files.
PLOTS_FOLDER_PATH_1 = '/home/pedrosantos/git/rl-data-distribution/analysis/plots/'
# Absolute path to folder containing plots files (second folder).
PLOTS_FOLDER_PATH_2 = '/home/pedrosantos/git/rl-data-distribution/analysis/plots/ilu_server/'
###########################################################################

ENVS = ['gridEnv1', 'gridEnv2', 'multiPathEnv', 'mountaincar', 'pendulum', 'cartpole']
ALGORITHMS = ['offline_dqn', 'offline_cql']
DATASET_TYPES = ['dirichlet', 'eps-greedy', 'boltzmann']
ALGO_COLORS= {'offline_dqn': '#ef553b', 'offline_cql': '#636efa'} # '#4fcc96'

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

# Build page layout.
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

# Runs picker.
runs_picker = html.Div(
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
                                        value='gridEnv1',
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
                                        value=['offline_dqn'],
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
                                        value=['eps-greedy', 'boltzmann'],
                                        multi=True,
                                        style={'color': '#3b505e'},
                                        className="dropdown-box-third",
                                    ),
                                    dcc.Checklist(
                                        id='dataset-coverage-picker',
                                        options=[{'label': 'force_coverage=False', 'value': False},
                                                {'label': 'force_coverage=True', 'value': True}],
                                        value=[False, True],
                                        inline=True,
                                        inputStyle={"margin-right": "5px", "margin-left": "20px"},
                                    )
                            ]
                        ),
                    ]),
            ])
page_layout.append(runs_picker)

# Graph1 section.
graph1_section = html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="eight columns",
                            children=[dcc.Graph(figure=None, id='graph1')]
                        ),
                        html.Div(
                            className="four columns",
                            children=[
                                    html.P("X-axis:"),
                                    dcc.Dropdown(
                                        id='graph1-x-axis-picker',
                                        options=[{'label': 'Dataset entropy', 'value': 'dataset_entropy'},
                                                {'label':'Dataset coverage', 'value': 'dataset_coverage'},
                                                {'label':'KL dist', 'value': 'kl_dist'},
                                                {'label':'Chi dist', 'value': 'chi_dist'}],
                                        value='dataset_entropy',
                                        multi=False,
                                        style={'color': '#3b505e'},
                                        className="dropdown-box-third",
                                    ),
                                    html.P("Y-axis:"),
                                    dcc.Dropdown(
                                        id='graph1-y-axis-picker',
                                        options=[{'label': 'Q-values error', 'value': 'qvals_avg_error'},
                                                {'label':'Rollouts rewards', 'value': 'rollouts_rewards_final'}],
                                        value='qvals_avg_error',
                                        multi=False,
                                        style={'color': '#3b505e'},
                                        className="dropdown-box-third",
                                    ),
                            ]
                        ),
                    ],
                )
point_info =  html.Div(className='row',
                    children=[
                        html.Div(id="graph1-point-info-text")
                    ])
images_section = html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="six columns",
                            children=[
                                    html.Div(id='graph1-img-1'),
                            ]
                        ),
                        html.Div(
                            className="six columns",
                            children=[
                                    html.Div(id='graph1-img-2'),
                            ]
                        ),
                    ],
                )
page_layout.append(html.Div(className="container", children=[graph1_section, point_info, images_section]))

app.layout = html.Div(
    style={"height": "100%"},
    children=page_layout,
)


@app.callback(
    Output("graph1", "figure"),
    [
        Input("env-picker", "value"),
        Input("algorithm-picker", "value"),
        Input("dataset-picker", "value"),
        Input("dataset-coverage-picker", "value"),
        Input("graph1-x-axis-picker", "value"),
        Input("graph1-y-axis-picker", "value"),
    ],
)
def callback_graph1(env, algorithms, dataset_types,
            dataset_force_coverage_types, x_axis, y_axis):

    if x_axis == "dataset_entropy":
        x_axis_lbl = "Dataset entropy"
    elif x_axis == "dataset_coverage":
        x_axis_lbl = "Dataset coverage"
    elif x_axis == "kl_dist":
        x_axis_lbl = "KL distance"
    elif x_axis == "chi_dist":
        x_axis_lbl = "Chi-square distance"
    else:
        raise ValueError("Unknown X-axis type.")

    if y_axis == "qvals_avg_error":
        y_axis_lbl = "Q-values error"
    elif y_axis == "rollouts_rewards_final":
        y_axis_lbl = "Reward"
    else:
        raise ValueError("Unknown Y-axis type.")

    layout = go.Layout(
        margin=go.layout.Margin(l=50, r=50, b=50, t=30),
        xaxis={"title": x_axis_lbl},
        yaxis={"title": y_axis_lbl},
        legend_orientation="h",
        legend=dict(y=-0.2)
    )

    traces = []

    for algo in algorithms:

        algo_data_X = []
        algo_data_Y = []
        hover_text = []
        exp_ids = []

        for dataset_type in dataset_types:

            for force_coverage_type in dataset_force_coverage_types:

                filtered_df = DATA.loc[
                                (DATA['env_id']==env) &
                                (DATA['algo_id']==algo) &
                                (DATA['dataset_type_id']==dataset_type) &
                                (DATA['force_dataset_coverage']==force_coverage_type)
                            ]

                algo_data_Y.extend(filtered_df[y_axis])
                algo_data_X.extend(filtered_df[x_axis])
                hover_text.extend(filtered_df['info_text'])
                exp_ids.extend(filtered_df['id'])

        traces.append(go.Scatter(
            x=algo_data_X,
            y=algo_data_Y,
            customdata=exp_ids,
            showlegend=True,
            marker=dict(
                color=ALGO_COLORS[algo],
                size=7,
                line=dict(width=1.5,
                    color='DarkSlateGrey'
                ),
            ),
            mode='markers',
            name=algo,
            hovertext=hover_text,
            hoverinfo="text",
        ))

    figure = go.Figure(data=traces, layout=layout)
    if y_axis == "qvals_avg_error":
        figure.update_layout(yaxis_type="log")
    if x_axis == "chi_dist":
        figure.update_layout(xaxis_type="log")

    return figure

@app.callback(
        [
        Output('graph1-img-1', 'children'),
        Output('graph1-img-2', 'children'),
        Output('graph1-point-info-text', 'children')
        ],
        [Input('graph1', 'clickData')])
def callback_graph1_point_images(selected_point):

    if not selected_point:
        raise PreventUpdate

    exp_id_to_plot = selected_point['points'][0]['customdata']

    # Load image.
    try:
        # Check if file exists and setup paths.
        if os.path.isfile(PLOTS_FOLDER_PATH_1 + exp_id_to_plot + '/scalar_metrics.json'):
            plots_folder_path = PLOTS_FOLDER_PATH_1
        elif os.path.isfile(PLOTS_FOLDER_PATH_2 + exp_id_to_plot + '/scalar_metrics.json'):
            plots_folder_path = PLOTS_FOLDER_PATH_2
        else:
            raise FileNotFoundError(f"Unable to find experiment {exp_id_to_plot} data.")

        img_1 = np.array(Image.open(plots_folder_path + exp_id_to_plot + '/q_values_summed_error.png'))
        img_2 = np.array(Image.open(plots_folder_path + exp_id_to_plot + '/rollouts_rewards.png'))

    except:
        raise PreventUpdate

    fig_1 = px.imshow(img_1)
    fig_1.update_layout(coloraxis_showscale=False)
    fig_1.update_xaxes(showticklabels=False)
    fig_1.update_yaxes(showticklabels=False)

    fig_2 = px.imshow(img_2)
    fig_2.update_layout(coloraxis_showscale=False)
    fig_2.update_xaxes(showticklabels=False)
    fig_2.update_yaxes(showticklabels=False)

    text_render = 'Point Info:\n' + selected_point['points'][0]['hovertext'].replace("<br>", "; ")

    return dcc.Graph(figure=fig_1), dcc.Graph(figure=fig_2), text_render

if __name__ == "__main__":
    app.run_server(debug=True)
