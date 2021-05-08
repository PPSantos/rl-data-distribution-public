from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots_single import main as plots
from analysis.plot_state_qval import main as Qvalplots


# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
VAL_ITER_DATA = 'gridEnv4_val_iter_2021-04-28-09-54-18'

EXP_IDS = [
# One-hot observations.
'gridEnv4_oracle_fqi_2021-04-29-17-59-57',
'gridEnv4_oracle_fqi_2021-04-29-18-26-04',
'gridEnv4_oracle_fqi_2021-04-29-18-50-15',
'gridEnv4_oracle_fqi_2021-04-29-19-14-41',
'gridEnv4_oracle_fqi_2021-04-29-19-39-33',
'gridEnv4_oracle_fqi_2021-04-29-20-04-27',
'gridEnv4_oracle_fqi_2021-04-29-20-29-14',

# Smoothed observations.
'gridEnv4_oracle_fqi_2021-04-29-20-54-26',
'gridEnv4_oracle_fqi_2021-04-29-21-18-38',
'gridEnv4_oracle_fqi_2021-04-29-21-42-55',
'gridEnv4_oracle_fqi_2021-04-29-22-08-03',
'gridEnv4_oracle_fqi_2021-04-29-22-33-17',
'gridEnv4_oracle_fqi_2021-04-29-22-58-29',
'gridEnv4_oracle_fqi_2021-04-29-23-23-37'
]

if __name__ == "__main__":

    for exp_id in EXP_IDS:
        print(exp_id)
        plots(exp_id, VAL_ITER_DATA)
        Qvalplots(exp_id, VAL_ITER_DATA)
