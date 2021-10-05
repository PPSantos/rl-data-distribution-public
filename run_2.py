import os
import json
import shutil
import tarfile
import pathlib
import numpy as np

from train import train
from train import DEFAULT_TRAIN_ARGS, VAL_ITER_DATA
from analysis.plots import main as plots

from utils.sampling_dist import DEFAULT_SAMPLING_DIST_ARGS
from utils.sampling_dist import main as sampling_dist
from algos.utils.array_functions import build_boltzmann_policy

#################################################################
ENV_NAME = 'gridEnv4'
NUM_SAMPLING_DISTS = 50
OPTIMAL_QVALS_EXP_ID = 'gridEnv4_offline_dqn_2021-10-03-00-02-12'
HIDDEN_LAYERS = {'gridEnv1': [20,40,20],
                 'gridEnv4': [20,40,20],
                 'multiPathsEnv': [20,40,20],
                 'pendulum': [32,64,32],
                 'mountaincar': [64,128,64]}
#################################################################

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/data'


if __name__ == "__main__":

    # Load args.
    args = DEFAULT_TRAIN_ARGS
    sampling_dist_args = DEFAULT_SAMPLING_DIST_ARGS
    algo_dict_key = args['algo'] + '_args'

    # Setup args.
    args['env_args']['env_name'] = ENV_NAME
    sampling_dist_args['env_args']['env_name'] = ENV_NAME
    val_iter_data = VAL_ITER_DATA[ENV_NAME]
    args[algo_dict_key]['hidden_layers'] = HIDDEN_LAYERS[ENV_NAME]

    # Load optimal Q-values.
    print(f'Loading optimal Q-values from exp {OPTIMAL_QVALS_EXP_ID}.')
    tar = tarfile.open(f"{DATA_FOLDER_PATH}/{OPTIMAL_QVALS_EXP_ID}.tar.gz")
    f = tar.extractfile(f"{OPTIMAL_QVALS_EXP_ID}/train_data.json")
    optimal_qvals_exp_data = json.loads(f.read())
    optimal_qvals_exp_data = json.loads(optimal_qvals_exp_data) # Why is this needed?
    qvals = np.array([e['Q_vals'] for e in optimal_qvals_exp_data]) # [R,(E),S,A]
    optimal_qvals = np.mean(qvals[:,-10:,:,:],axis=1) #[R,S,A]
    print(optimal_qvals.shape)

    exp_ids = []
    sampling_dist_ids = []
    for _ in range(NUM_SAMPLING_DISTS):

        # Build sampling policy.
        run_idx = np.random.choice(np.arange(optimal_qvals.shape[0])) # Randomly select the Q-values of one of the runs.
        run_qvals = optimal_qvals[run_idx,:,:] # [S,A]
        policy = build_boltzmann_policy(run_qvals, temperature=np.random.uniform(low=-10.0,high=10.0))

        # Create sampling dist.
        sampling_dist_path, sampling_dist_id = sampling_dist(policy=policy, args=sampling_dist_args)
        sampling_dist_ids.append(sampling_dist_id)

        # Run ofline RL.
        args[algo_dict_key]['dataset_custom_sampling_dist'] = sampling_dist_path
        exp_path, exp_id = train(args)
        exp_ids.append(exp_id)
        # Compute plots.
        plots(exp_id, val_iter_data)
        # Compress and cleanup.
        shutil.make_archive(exp_path,
                            'gztar',
                            os.path.dirname(exp_path),
                            exp_id)
        shutil.rmtree(exp_path)

        print('Exp. ids:', exp_ids)
        print('Sampling dist. ids:', sampling_dist_ids)
