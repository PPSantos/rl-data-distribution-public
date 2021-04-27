from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots_single import main as plots
from analysis.plot_state_qval import main as Qvalplots


# `VAL_ITER_DATA` must be set if `COMPUTE_PLOTS` is True.
VAL_ITER_DATA = '8_8_val_iter_2021-04-13-17-55-56'

EXP_IDS = [
    # One-hot observations.
    '8_8_oracle_fqi_2021-04-27-00-33-38',
    '8_8_oracle_fqi_2021-04-27-00-43-49',
    '8_8_oracle_fqi_2021-04-27-00-53-59',
    '8_8_oracle_fqi_2021-04-27-01-03-40',
    '8_8_oracle_fqi_2021-04-27-01-13-34',
    '8_8_oracle_fqi_2021-04-27-01-23-29',
    '8_8_oracle_fqi_2021-04-27-01-33-12',

    # Smoothed observations.
    '8_8_oracle_fqi_2021-04-27-01-43-11',
    '8_8_oracle_fqi_2021-04-27-01-53-03',
    '8_8_oracle_fqi_2021-04-27-02-02-48',
    '8_8_oracle_fqi_2021-04-27-02-12-35',
    '8_8_oracle_fqi_2021-04-27-02-22-15',
    '8_8_oracle_fqi_2021-04-27-02-32-03',
    '8_8_oracle_fqi_2021-04-27-02-41-40',

    # Random observations.
    '8_8_oracle_fqi_2021-04-27-02-51-30',
    '8_8_oracle_fqi_2021-04-27-03-01-09',
    '8_8_oracle_fqi_2021-04-27-03-10-55',
    '8_8_oracle_fqi_2021-04-27-03-20-44',
    '8_8_oracle_fqi_2021-04-27-03-30-26',
    '8_8_oracle_fqi_2021-04-27-03-40-12',
    '8_8_oracle_fqi_2021-04-27-03-49-59'
]


if __name__ == "__main__":

    for exp_id in EXP_IDS:
        print(exp_id)
        #plots(exp_id, VAL_ITER_DATA)
        Qvalplots(exp_id, VAL_ITER_DATA)
