from train import train
from train import DEFAULT_TRAIN_ARGS
from analysis.plots import main as plots

EXP_IDS = [
    'gridEnv1_dqn_e_tab_2021-07-15-19-48-25',
]

VAL_ITER_ID = 'gridEnv1_val_iter_2021-05-14-15-54-10'

if __name__ == "__main__":

    for e_id in EXP_IDS:
        plots(e_id, VAL_ITER_ID)
