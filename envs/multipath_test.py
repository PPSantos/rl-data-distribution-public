from envs.multipath import MultiPathEnv, MultiPathRandomObservation

import numpy as np

if __name__ == "__main__":

    # env = MultiPathEnv()
    # obs = env.reset()
    # env.render()
    # done = False
    # while not done:
    #     a = np.random.choice([0,1,2,3,4])
    #     print('a=', a)
    #     next_obs, reward, done, info = env.step(a)
    #     print('r=', reward)
    #     env.render()

    env = MultiPathRandomObservation(features_dim=8)
    obs = env.reset()
    print(obs)
    env.render()
    done = False
    while not done:
        a = np.random.choice([0,1,2,3,4])
        print('a=', a)
        next_obs, reward, done, info = env.step(a)
        print('r=', reward)
        print(next_obs)
        env.render()
