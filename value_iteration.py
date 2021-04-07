import sys
import numpy as np

from envs import env_suite

SIZE_X = 16
SIZE_Y = 16

GAMMA = 0.95
EPSILON = 0.05

def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError()

if __name__ == "__main__":

    env = env_suite.random_grid_env(size_x=SIZE_X, size_y=SIZE_Y, dim_obs=16, time_limit=50, wall_ratio=0.2, seed=0,
                                    tabular=True, smooth_obs=False, one_hot_obs=False, distance_reward=False, absorb=False)

    print('Env num states:', env.num_states)
    print('Env num actions:', env.num_actions)
    print('Transition matrix shape:', env.transition_matrix().shape)
    print('Initial state distribution:', env.initial_state_distribution)
    print('\n')

    env.render()

    """
        Value Iteration.
    """
    Q_vals = np.zeros((env.num_states, env.num_actions))
    
    for _ in range(500):
        Q_vals_old = np.copy(Q_vals)
        
        for state in range(env.num_states):
            for action in range(env.num_actions):

                Q_vals[state][action] = env.reward(state, 0, 0) + \
                            GAMMA * np.dot(env.transition_matrix()[state][action], np.max(Q_vals, axis=1))

        delta = np.sum(np.abs(Q_vals - Q_vals_old))
        print('delta:', delta)

        if delta < EPSILON:
            break

    # Calculate policy.
    policy = {}
    max_Q_vals = {}
    for state in range(env.num_states):
        policy[state] = np.argmax(Q_vals[state])
        max_Q_vals[state] = np.max(Q_vals[state])

    print('Q_vals:', max_Q_vals)
    print('Policy:', policy)
    env.render()

    # Plot policy.
    sys.stdout.write('-'*(SIZE_X+2)+'\n')
    for h in range(SIZE_Y):
        sys.stdout.write('|')
        for w in range(SIZE_X):
            sys.stdout.write(str(policy[xy_to_idx((w,h), SIZE_X, SIZE_Y)]))
        sys.stdout.write('|\n')
    sys.stdout.write('-' * (SIZE_X + 2)+'\n')

    # Plot max Q-values.
    for h in range(SIZE_Y):
        for w in range(SIZE_X):
            sys.stdout.write("{:.1f} ".format(max_Q_vals[xy_to_idx((w,h), SIZE_X, SIZE_Y)]))
        sys.stdout.write('\n')
    sys.stdout.write('\n')
