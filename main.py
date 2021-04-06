import numpy as np

from rlutil.envs.tabular_cy import tabular_env

from envs import env_suite


if __name__ == "__main__":

    env = env_suite.random_grid_env(size_x=16, size_y=16, dim_obs=16, time_limit=50, wall_ratio=0.0, seed=0,
                                    tabular=True, smooth_obs=False, one_hot_obs=False, distance_reward=False, absorb=False)

    print('Env num states:', env.num_states)
    print('Env num actions:', env.num_actions)
    print('Transition matrix shape:', env.transition_matrix().shape)
    print('Initial state distribution:', env.initial_state_distribution)
    print('\n')

    aspace = env.action_space
    #print(env.reward())
    obs = env.reset()

    t = 0
    for _ in range(5000):

        print(f"================== {t} ==================")
        env.render()
        print('obs', obs)

        # a = aspace.sample()
        a = np.random.choice([1,4])
        print('action:', a)

        obs, reward, done, info = env.step(a)
        print('reward', reward)
        print('done', done)
        print('info', info)

        t += 1
        if done:
            break

        # state = env.get_state()
        # print('s:', state)
