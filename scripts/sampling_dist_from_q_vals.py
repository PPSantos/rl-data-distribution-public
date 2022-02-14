"""
    This script loads the final Q-values from a given experiment and calculates
    the (s,a) distributions (sampling dists.) induced by an eps-greedy policy
    w.r.t. the Q-values.
"""
import os
import json
import tarfile
import pathlib
import numpy as np
from tqdm import tqdm

from scripts.dataset import _calculate_dataset_dist_from_counts
from utils.array_functions import build_eps_greedy_policy
from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder
from envs import env_suite


#################################################################
ENV_NAME = 'pendulum'
# QVALS_EXP_ID = 'gridEnv1_offline_dqn_2022-02-06-02-14-43'
# QVALS_EXP_ID = 'gridEnv2_offline_dqn_2022-01-22-15-38-44'
# QVALS_EXP_ID = 'multiPathEnv_offline_dqn_2022-01-24-03-06-12'
# QVALS_EXP_ID = 'mountaincar_offline_dqn_2022-02-04-00-08-15'
QVALS_EXP_ID = 'pendulum_offline_dqn_2022-02-11-02-30-50'
EPSILON = 0.0
#################################################################

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data'

def rollout_policy(env, env_grid_spec, policy, num_episodes=1_000):

    # Rollout policy.
    episode_rewards = []

    # Matrix to store (s,a) counts.
    sa_counts = np.zeros((env.num_states, env.num_actions))

    for _ in tqdm(range(num_episodes)):

        obs = env.reset()
        s_t = env.get_state()

        done = False
        episode_cumulative_reward = 0
        while not done:

            # Pick action.
            a_t = policy(s_t)

            # Env step.
            obs_t1, r_t1, done, info = env.step(a_t)
            s_t1 = env.get_state()

            # Log data.
            episode_cumulative_reward += r_t1
            sa_counts[s_t,a_t] += 1

            s_t = s_t1

        episode_rewards.append(episode_cumulative_reward)

    print('Policy average reward:', np.mean(episode_rewards))

    s_dist, sa_counts = _calculate_dataset_dist_from_counts(env, env_grid_spec, sa_counts)

    return s_dist


def main():

    # Setup experiment data folder.
    exp_name = create_exp_name({'env_name': ENV_NAME, 'algo': 'sampling_dist'})
    exp_path = DATA_FOLDER_PATH + f"/{exp_name}"
    os.makedirs(exp_path, exist_ok=True)
    print('\nSampling dist. ID:', exp_name)

    # Load environment.
    env, env_grid_spec = env_suite.get_env(ENV_NAME)

    # Load optimal Q-values.
    print(f'Loading Q-values from exp {QVALS_EXP_ID}.')
    tar = tarfile.open(f"{DATA_FOLDER_PATH}/{QVALS_EXP_ID}.tar.gz")
    f = tar.extractfile(f"{QVALS_EXP_ID}/train_data.json")
    qvals_exp_data = json.loads(f.read())
    qvals_exp_data = json.loads(qvals_exp_data) # Why is this needed?
    qvals = np.array([e['Q_vals'] for e in qvals_exp_data]) # [R,(E),S,A]

    final_qvals = qvals[:,-1,:,:] #[R,S,A]
    print(final_qvals.shape)

    # Calculate (s,a) dists. induced by the eps-greedy
    # policies w.r.t. the final Q-values.
    sampling_dists = []
    for run_idx in range(final_qvals.shape[0]):
        run_qvals = final_qvals[run_idx,:,:] # [S,A]
        policy = build_eps_greedy_policy(run_qvals, epsilon=EPSILON)

        # Rollout policy and retrieve sampling dist.
        s_dist = rollout_policy(env, env_grid_spec, policy=policy)
        sampling_dists.append(s_dist)

    # Store data.
    data = {}
    data['sampling_dists'] =  sampling_dists
    f = open(exp_path + "/data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()
    
    print('Sampling dist. ID:', exp_name)

if __name__ == "__main__":
    main()
