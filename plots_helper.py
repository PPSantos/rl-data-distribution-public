from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots_single import main as plots
from analysis.plot_state_qval import main as Qvalplots


# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
VAL_ITER_DATA = '8_8_val_iter_2021-04-13-17-55-56'

EXP_IDS = [
    # One-hot observations.
    '8_8_oracle_fqi_2021-04-26-15-48-35',
    '8_8_oracle_fqi_2021-04-26-15-58-11',
    '8_8_oracle_fqi_2021-04-26-16-07-38',
    '8_8_oracle_fqi_2021-04-26-16-16-54',
    '8_8_oracle_fqi_2021-04-26-16-26-13',
    '8_8_oracle_fqi_2021-04-26-16-35-30',
    '8_8_oracle_fqi_2021-04-26-16-44-54',
    
    # Smoothed observations.
    '8_8_oracle_fqi_2021-04-26-16-54-17',
    '8_8_oracle_fqi_2021-04-26-17-03-33',
    '8_8_oracle_fqi_2021-04-26-17-12-58',
    '8_8_oracle_fqi_2021-04-26-17-22-23',
    '8_8_oracle_fqi_2021-04-26-17-31-40',
    '8_8_oracle_fqi_2021-04-26-17-41-00',
    '8_8_oracle_fqi_2021-04-26-17-50-06',

    # Random observations.
    '8_8_oracle_fqi_2021-04-26-17-59-14',
    '8_8_oracle_fqi_2021-04-26-18-08-22',
    '8_8_oracle_fqi_2021-04-26-18-17-39',
    '8_8_oracle_fqi_2021-04-26-18-27-00',
    '8_8_oracle_fqi_2021-04-26-18-36-04',
    '8_8_oracle_fqi_2021-04-26-18-45-18',
    '8_8_oracle_fqi_2021-04-26-18-54-28'
]

if __name__ == "__main__":

    for exp_id in EXP_IDS:
        print(exp_id)
        #plots(exp_id, VAL_ITER_DATA)
        Qvalplots(exp_id, VAL_ITER_DATA)
