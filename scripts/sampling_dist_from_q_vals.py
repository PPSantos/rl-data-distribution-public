"""
    This script loads the final Q-values from a given experiment and calculates
    the (s,a) distributions (sampling dists.) induced by an eps-greedy policy
    w.r.t. the Q-values.
"""
import json
import tarfile
import pathlib
import numpy as np

from scripts.sampling_dist import main as sampling_dist
from utils.array_functions import build_eps_greedy_policy

#################################################################
ENV_NAME = 'gridEnv1'
QVALS_EXP_ID = 'gridEnv1_offline_dqn_2022-01-16-13-09-39'
EPSILON = 0.1
#################################################################

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data'

def main():

    # Load optimal Q-values.
    print(f'Loading Q-values from exp {QVALS_EXP_ID}.')
    tar = tarfile.open(f"{DATA_FOLDER_PATH}/{QVALS_EXP_ID}.tar.gz")
    f = tar.extractfile(f"{QVALS_EXP_ID}/train_data.json")
    qvals_exp_data = json.loads(f.read())
    qvals_exp_data = json.loads(qvals_exp_data) # Why is this needed?
    qvals = np.array([e['Q_vals'] for e in qvals_exp_data]) # [R,(E),S,A]
    final_qvals = np.mean(qvals[:,-10:,:,:],axis=1) #[R,S,A]
    print(final_qvals.shape)

    # Calculate (s,a) dist. induced by the greedy
    # policy w.r.t. the final Q-values.
    sampling_dist_ids = []
    sampling_dist_paths = []
    for run_idx in range(final_qvals.shape[0]):
        run_qvals = final_qvals[run_idx,:,:] # [S,A]
        policy = build_eps_greedy_policy(run_qvals, epsilon=EPSILON)

        # Rollout policy and retrieve sampling dist.
        sampling_dist_path, sampling_dist_id = sampling_dist(env_name=ENV_NAME, policy=policy)
        sampling_dist_ids.append(sampling_dist_id)
        sampling_dist_paths.append(sampling_dist_path)

        print('Sampling dist. ids:', sampling_dist_ids)
        print('Sampling dist. paths:', sampling_dist_paths)

if __name__ == "__main__":
    main()
