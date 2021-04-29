from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots_single import main as plots
from analysis.plot_state_qval import main as Qvalplots


# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
VAL_ITER_DATA = 'gridEnv4_val_iter_2021-04-28-09-54-18'

EXP_IDS = ['gridEnv4_oracle_fqi_2021-04-28-14-50-24',
'gridEnv4_oracle_fqi_2021-04-28-15-10-32',
'gridEnv4_oracle_fqi_2021-04-28-15-29-45',
'gridEnv4_oracle_fqi_2021-04-28-15-49-23',
'gridEnv4_oracle_fqi_2021-04-28-16-09-00',
'gridEnv4_oracle_fqi_2021-04-28-16-28-37',
'gridEnv4_oracle_fqi_2021-04-28-16-47-55',
'gridEnv4_oracle_fqi_2021-04-28-17-07-24',
'gridEnv4_oracle_fqi_2021-04-28-17-27-00',
'gridEnv4_oracle_fqi_2021-04-28-17-47-07',
'gridEnv4_oracle_fqi_2021-04-28-18-07-07',
'gridEnv4_oracle_fqi_2021-04-28-18-27-25',
'gridEnv4_oracle_fqi_2021-04-28-18-47-34',
'gridEnv4_oracle_fqi_2021-04-28-19-07-41',
'gridEnv4_oracle_fqi_2021-04-28-19-27-40',
'gridEnv4_oracle_fqi_2021-04-28-19-47-47',
'gridEnv4_oracle_fqi_2021-04-28-20-07-51',
'gridEnv4_oracle_fqi_2021-04-28-20-27-56',
'gridEnv4_oracle_fqi_2021-04-28-20-48-05',
'gridEnv4_oracle_fqi_2021-04-28-21-08-22',
'gridEnv4_oracle_fqi_2021-04-28-21-28-33']

if __name__ == "__main__":

    for exp_id in EXP_IDS:
        print(exp_id)
        #plots(exp_id, VAL_ITER_DATA)
        Qvalplots(exp_id, VAL_ITER_DATA)
