from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots import main as plots

EXP_IDS = [
    'gridEnv4_dqn2be_2021-07-14-16-26-32',
]

VAL_ITER_ID = 'gridEnv4_val_iter_2021-06-16-10-08-44'

if __name__ == "__main__":

    for e_id in EXP_IDS:
        plots(e_id, VAL_ITER_ID)
