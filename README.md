# RL-data-distribution

### Installation
1) Setup and activate virtual environment (tested with python 3.8.10)
2) Git clone this repo: ``` git clone https://github.com/PPSantos/rl-data-distribution.git```
3) ``` cd rl-data-distribution```
4) Install requirements: ``` pip install -r requirements.txt ```
5) Clone d3rlpy library: ``` git clone https://github.com/takuseno/d3rlpy.git ```
6) Detach library's HEAD: ```cd d3rlpy``` + ```git checkout f20f1e662e1c22daaa7b2bc9c8cd4771fd2965f8```
7) Install d3rlpy library: ``` pip install -e .```
8) ``` cd ..```
9) Test installation: ``` python scripts/run.py```

### Run dashboard
1) Setup ```analysis/dash_app/app.py``` global path variables: ```CSV_PATH, PLOTS_FOLDER_PATH_1, PLOTS_FOLDER_PATH_2```
2) Run dashboard: ```python analysis/dash_app/app.py```
