import pathlib

import dash
from dash import dcc
from dash import html
# import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import numpy as np
import pandas as pd

# from analysis.plots_appendix import PLOTS_FOLDER_PATH

# Absolute path to folder containing experiments data.
CSV_PATH = '/home/pedrosantos/git/rl-data-distribution/data/parsed_data.csv'

PLOTS_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + '/analysis/plots/'

ENVS = ['GridEnv1', 'GridEnv2'] # 'MultiPath']
ALGORITHMS = ['DQN', 'CQL']
DATASET_TYPES = ['Dirichlet', 'Eps-greedy', 'Boltzmann']

DATA = pd.read_csv(CSV_PATH)

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

ALGO_COLORS= {'DQN': '#d62728', 'CQL': '#1f77b4'}
# COLORS = ['#ff7f0e', '#2ca02c', '#9467bd']


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
graph_section = html.Div(
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
                                                {'label':'Dataset coverage', 'value': 'dataset_coverage'}],
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
                                    html.P("Point info:"),
                                    html.Div(id='graph1-point-info')
                            ]
                        ),
                    ],
                )
page_layout.append(html.Div(className="container", children=[graph_section]))


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
        Input("graph1-x-axis-picker", "value"),
        Input("graph1-y-axis-picker", "value"),
    ],
)
def callback_graph1(env, algorithms, dataset_types, x_axis, y_axis):

    if x_axis == "dataset_entropy":
        x_axis_lbl = "Dataset entropy"
    elif x_axis == "dataset_coverage":
        x_axis_lbl = "Dataset coverage"
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

        for dataset_type in dataset_types:

            filtered_df = DATA.loc[(DATA['env_id']==env) & (DATA['algo_id']==algo)
                        & (DATA['dataset_type_id']==dataset_type)]

            algo_data_Y.extend(filtered_df[y_axis])
            algo_data_X.extend(filtered_df[x_axis])
            hover_text.extend(filtered_df['info_text'])

        traces.append(go.Scatter(
            x=algo_data_X,
            y=algo_data_Y,
            showlegend=True,
            marker_color=ALGO_COLORS[algo],
            mode='markers',
            name=algo,
            hovertext=hover_text,
            hoverinfo="text",
        ))

    figure = go.Figure(data=traces, layout=layout)
    if y_axis == "qvals_avg_error":
        figure.update_layout(yaxis_type="log")

    return figure

@app.callback(
        Output('graph1-point-info', 'children'),
        [Input('graph1', 'clickData')])
def callback_graph1_point(selection):
    print(selection)
    if selection is not None:
        # x_data = np.linspace(0,500,500)
        # y_data = np.random.rand(500)
        
        # # depending on the station text use different color for line
        # if selection['points'][0]['text'] == 'red':
        #     color='#ff0000'
        # else:
        #     color='#0000ff'
        # data = [go.Scatter(
        #             x=x_data,
        #             y=y_data,
        #             line={'color': color},
        #             opacity=0.8,
        #             name="Graph"
        #         )]
        # layout = go.Layout(
        #             xaxis={'type': 'linear', 'title': "Timestep"},
        #             yaxis={'type': 'linear', 'title': "Value"},
        #             margin={'l': 60, 'b': 40, 'r': 10, 't': 10},
        #             hovermode="False"
        #             )
        
        # return {'data': data, 'layout': layout}
        return []

if __name__ == "__main__":
    app.run_server(debug=True)
