"""
    Experimental setup 3: Vary the (statistical) distance between the sampling distribution
    and the closest distribution induced by one of the optimal policies of the MDP.

    This is achieved by first loading the optimal Q-values and then generating sampling
    distributions using a Boltzmann policy with a temperature coefficient varying in the
    interval [-10 (anti-optimal), 10 (optimal)].
"""
import os
import json
import shutil
import tarfile
import pathlib
import numpy as np
import scipy

from train import train
from train import DEFAULT_TRAIN_ARGS, VAL_ITER_DATA
from analysis.plots import main as plots

from scripts.sampling_dist import main as sampling_dist
from utils.array_functions import build_boltzmann_policy

#################################################################
ENV_NAME = 'gridEnv1'
NUM_SAMPLING_DISTS = 25
OPTIMAL_QVALS_EXP_ID = 'gridEnv1_offline_dqn_2022-01-16-13-09-39'
HIDDEN_LAYERS = {'gridEnv1': [20,40,20],
                 'gridEnv4': [20,40,20],
                 'multiPathsEnv': [20,40,20],
                 'pendulum': [32,64,32],
                 'mountaincar': [64,128,64]}

# Additional rejection sampling args.
# (if `OPTIMAL_SAMPLING_DIST_IDS` is not None).
OPTIMAL_SAMPLING_DIST_IDS = None
TARGET_INTERVAL = [2,10]
#################################################################

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data'


if __name__ == "__main__":

    # Load args.
    args = DEFAULT_TRAIN_ARGS
    algo_dict_key = args['algo'] + '_args'

    # Setup train args.
    args['env_name'] = ENV_NAME
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

    if OPTIMAL_SAMPLING_DIST_IDS is not None:
        print('Loading optimal sampling dists. (rejection sampling).')
        optimal_dists = []
        for optimal_dist_id in OPTIMAL_SAMPLING_DIST_IDS:
            optimal_dist_path = DATA_FOLDER_PATH + '/' + optimal_dist_id + '/data.json'
            print(optimal_dist_path)
            with open(optimal_dist_path, 'r') as f:
                data = json.load(f)
                d = np.array(json.loads(data)['sampling_dist'])
            f.close()
            optimal_dists.append(d)
        print('len(optimal_dists)', len(optimal_dists))

    exp_ids = []
    sampling_dist_ids = []
    num_sampled_dists = 0
    while num_sampled_dists < NUM_SAMPLING_DISTS:
    
        # Build sampling policy.
        run_idx = np.random.choice(np.arange(optimal_qvals.shape[0])) # Randomly select the Q-values of one of the runs.
        run_qvals = optimal_qvals[run_idx,:,:] # [S,A]
        sampled_temperature = np.random.uniform(low=-10.0, high=10.0)
        print('sampled_temperature', sampled_temperature)
        policy = build_boltzmann_policy(run_qvals, temperature=sampled_temperature)

        # Create sampling dist.
        sampling_dist_path, sampling_dist_id = sampling_dist(env_name=ENV_NAME, policy=policy)

        if OPTIMAL_SAMPLING_DIST_IDS is not None:
            # Only accept sampled dist if distance is inside `TARGET_INTERVAL`.
            with open(sampling_dist_path, 'r') as f:
                data = json.load(f)
                sampled_dist = np.array(json.loads(data)['sampling_dist'])
            f.close()
            print(sampling_dist_path)
            kl_dist = np.min([scipy.stats.entropy(optimal_dist, sampled_dist+1e-06)
                            for optimal_dist in optimal_dists])
            print('kl_dist', kl_dist)

            if (kl_dist < TARGET_INTERVAL[0]) or (kl_dist > TARGET_INTERVAL[1]):
                print('REJECTED')
                shutil.rmtree(pathlib.Path(sampling_dist_path).parent.absolute())
                continue

        sampling_dist_ids.append(sampling_dist_id)
        num_sampled_dists += 1
        print('ACCEPTED')

        # Run ofline RL.
        args[algo_dict_key]['dataset_custom_sampling_dist'] = sampling_dist_path
        exp_path, exp_id = train(args)
        exp_ids.append(exp_id)
        # Compute plots.
        plots(exp_id, VAL_ITER_DATA[ENV_NAME])
        # Compress and cleanup.
        shutil.make_archive(exp_path,
                            'gztar',
                            os.path.dirname(exp_path),
                            exp_id)
        shutil.rmtree(exp_path)

        print('Exp. ids:', exp_ids)
        print('Sampling dist. ids:', sampling_dist_ids)
