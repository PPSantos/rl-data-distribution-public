"""
    This script loads the final Q-values from a given experiment and
    calculates the (s,a) distribution (sampling dist. induced by the
    Q-values.
"""
import os
import json
import shutil
import tarfile
import pathlib
import numpy as np

from utils.sampling_dist import main as sampling_dist
from utils.array_functions import build_eps_greedy_policy

#################################################################
ENV_NAME = 'multiPathsEnv'
QVALS_EXP_ID = 'multiPathsEnv_offline_dqn_2021-10-03-11-15-09'
EPSILON = 0.25
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
    # print(final_qvals.shape)

    # Calculate (s,a) dist. induced by the greedy
    # policy w.r.t. the final Q-values.
    sampling_dist_ids = []
    for run_idx in range(final_qvals.shape[0]):
        run_qvals = final_qvals[run_idx,:,:] # [S,A]
        policy = build_eps_greedy_policy(run_qvals, epsilon=EPSILON)

        # Rollout policy and retrieve sampling dist.
        _, sampling_dist_id = sampling_dist(env_name=ENV_NAME, policy=policy)
        sampling_dist_ids.append(sampling_dist_id)

        print('Sampling dist. ids:', sampling_dist_ids)

if __name__ == "__main__":
    main()
