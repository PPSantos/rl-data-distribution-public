"""
    Script that calculates the sampling distribution induced by a given policy.
"""
import os
import json
import numpy as np
import pathlib
from tqdm import tqdm
import scipy.stats

from envs import env_suite
from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'


def main(env_name, policy, num_episodes=1_000):

    # Setup experiment data folder.
    exp_name = create_exp_name({'env_name': env_name, 'algo': 'sampling_dist'})
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nSampling dist. ID:', exp_name)
    print('scripts/sampling_dist.py arguments:')
    print('env_name:', env_name, ', num_episodes:', num_episodes)

    # Load environment.
    env, _ = env_suite.get_env(env_name, seed=0)

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

    data = {}
    data['episode_rewards'] = episode_rewards
    data['sa_counts'] = sa_counts # [S,A]

    sampling_dist = sa_counts / np.sum(sa_counts) # [S,A]
    sampling_dist_flattened = sampling_dist.flatten() # [S*A]
    print('(S,A) dist. entropy:', scipy.stats.entropy(sampling_dist_flattened))
    data['sampling_dist'] = sampling_dist_flattened

    # Store data.
    f = open(exp_path + "/data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return exp_path + "/data.json", exp_name
