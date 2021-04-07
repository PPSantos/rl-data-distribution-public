import sys
import numpy as np

from algos import q_learning, value_iteration
from envs import env_suite

# Env. arguments:
size_x = 5
size_y = 5
seed = 14

# General arguments.
algo = 'val_iter'
gamma = 0.95
num_episodes = 50000

# Value iteration arguments.
val_iter_epsilon = 0.05

# Q-learning arguments.
alpha = 0.1
expl_eps = 0.2


def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError()


if __name__ == "__main__":

    env = env_suite.random_grid_env(size_x=size_x, size_y=size_y, dim_obs=16,
                                    time_limit=50, wall_ratio=0.0, seed=seed,
                                    tabular=True, smooth_obs=False, one_hot_obs=False,
                                    distance_reward=False, absorb=False)

    print('Env num states:', env.num_states)
    print('Env num actions:', env.num_actions)
    print('Transition matrix shape:', env.transition_matrix().shape)
    print('Initial state distribution:', env.initial_state_distribution)
    print('Env render:')
    env.render()
    print('\n')

    if algo == 'val_iter':
        agent = value_iteration.ValueIteration(env, gamma, val_iter_epsilon)
    elif algo == 'q_learning':
        agent = q_learning.QLearning(env, alpha, gamma, expl_eps)
    else:
        raise ValueError("Unknown algorithm.")

    Q_vals, max_Q_vals, policy = agent.train(num_episodes=num_episodes)

    print('\nQ-vals:', Q_vals)
    print('Max Q-vals:', max_Q_vals)
    print('Policy:', policy)
    env.render()

    # Plot policy.
    print('Policy:')
    sys.stdout.write('-'*(size_x+2)+'\n')
    for h in range(size_y):
        sys.stdout.write('|')
        for w in range(size_x):
            sys.stdout.write(str(policy[xy_to_idx((w,h), size_x, size_y)]))
        sys.stdout.write('|\n')
    sys.stdout.write('-' * (size_x + 2)+'\n')

    # Plot max Q-values.
    print('Max Q-vals:')
    for h in range(size_y):
        for w in range(size_x):
            sys.stdout.write("{:.1f} ".format(max_Q_vals[xy_to_idx((w,h), size_x, size_y)]))
        sys.stdout.write('\n')
    sys.stdout.write('\n')
