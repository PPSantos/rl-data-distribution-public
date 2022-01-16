from envs.grid_env import GridEnv, GridEnvRandomObservation
from envs import grid_spec

import numpy as np

if __name__ == "__main__":

    gs = grid_spec.spec_from_sparse_locations(8, 8,
                            {grid_spec.START: [(0, 7)],
                            grid_spec.WALL: [],
                            grid_spec.REWARD: [(7, 0)]})

    # env = GridEnv(gs)
    # obs = env.reset()
    # env.render()
    # done = False
    # while not done:
    #     a = np.random.choice([1,4])
    #     print('a=', a)
    #     next_obs, reward, done, info = env.step(a)
    #     print('r=', reward)
    #     env.render()

    env = GridEnvRandomObservation(obs_dim=8, gridspec=gs)
    obs = env.reset()
    print(obs)
    env.render()
    done = False
    while not done:
        a = np.random.choice([1,4])
        print('a=', a)
        next_obs, reward, done, info = env.step(a)
        print('r=', reward)
        print(next_obs)
        env.render()
