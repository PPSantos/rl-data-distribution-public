from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots_single import main as plots
from analysis.plot_state_qval import main as Qvalplots


# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
VAL_ITER_DATA = 'gridEnv4_val_iter_2021-04-28-09-54-18'

EXP_IDS = [
# One-hot observations.    
'gridEnv4_oracle_fqi_2021-04-28-23-55-32',
'gridEnv4_oracle_fqi_2021-04-29-00-17-56',
'gridEnv4_oracle_fqi_2021-04-29-00-40-23',
'gridEnv4_oracle_fqi_2021-04-29-01-03-29',
'gridEnv4_oracle_fqi_2021-04-29-01-26-22',
'gridEnv4_oracle_fqi_2021-04-29-01-49-16',
'gridEnv4_oracle_fqi_2021-04-29-02-12-01',

# Smoothed observations.    
'gridEnv4_oracle_fqi_2021-04-29-02-34-52',
'gridEnv4_oracle_fqi_2021-04-29-02-59-34',
'gridEnv4_oracle_fqi_2021-04-29-03-25-15',
'gridEnv4_oracle_fqi_2021-04-29-03-50-03',
'gridEnv4_oracle_fqi_2021-04-29-04-14-35',
'gridEnv4_oracle_fqi_2021-04-29-04-39-05',
'gridEnv4_oracle_fqi_2021-04-29-05-03-42']

if __name__ == "__main__":

    for exp_id in EXP_IDS:
        print(exp_id)
        plots(exp_id, VAL_ITER_DATA)
        # Qvalplots(exp_id, VAL_ITER_DATA)
