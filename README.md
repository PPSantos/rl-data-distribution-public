# The impact of data distribution on Q-learning with function approximation

This repository contains the code for the experimental results presented in: https://arxiv.org/abs/2111.11758

## Installation
1) Setup and activate virtual environment (tested with python 3.8.10)
2) Git clone this repo: ``` git clone https://github.com/PPSantos/rl-data-distribution.git```
3) ``` cd rl-data-distribution```
4) Install requirements: ``` pip install -r requirements.txt ```
5) ``` pip install -e . ```
9) Test installation: ``` python scripts/run.py```

## Four-state MDP

Code for the four-state MDP experiments:
- ```four_state_mdp/offline_plots.py``` computes the plots for the (offline + oracle version) and (offline + TD Error version) versions of the experiments.
- TODO


## Empirically assessing the impact of the data distribution in offline RL

Code for Sec. 5 of the paper:
- ``` scripts/run.py```: contains the main script that generates a dataset (using ```scripts/dataset.py```) and feeds it to the RL algorithm to train (using ```scripts/train.py```). Check the ``` RUN_ARGS ``` global variable in ``` scripts/run.py ``` for a description of the different dataset types and options available, as well as the different RL algorithms and respective hyperparameters.

Compiled data:
- ``` parsed_results.csv ```: contains all the compiled data used to produce the article's plots. 

Run dashboard:
 1) Setup ```analysis/dash_app/app.py``` global ```CSV_PATH``` path variable to point to the ``` parsed_results.csv ``` file.
 2) Run dashboard: ```python analysis/dash_app/app.py```
