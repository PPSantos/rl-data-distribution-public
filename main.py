import sys
import numpy as np

from algos import value_iteration, q_learning, dqn
from envs import env_suite

# General arguments.
args = {
    'algo': 'dqn',
    'num_episodes': 10_000,
    'gamma': 0.9,
}

# Env. arguments:
env_args = {
    'size_x': 8,
    'size_y': 8,
    'dim_obs': 8,
    'time_limit': 50,
    'wall_ratio': 0.0,
    'tabular': False,
    'seed': 421,
}

# Value iteration arguments.
val_iter_args = {
    'epsilon': 0.05
}

# Q-learning arguments.
q_learning_args = {
    'alpha': 0.05,
    'expl_eps_init': 0.9,
    'expl_eps_final': 0.01,
    'expl_eps_episodes': 9_000,
}

# DQN arguments.
dqn_args = {
    'batch_size': 128,
    'prefetch_size': 4,
    'target_update_period': 5_000,
    'samples_per_insert': 16.0,
    'min_replay_size': 50_000,
    'max_replay_size': 500_000,
    'importance_sampling_exponent': 0.9,
    'priority_exponent': 0.6,
    'n_step': 1,
    'epsilon_init': 0.9,
    'epsilon_final': 0.01,
    'epsilon_schedule_timesteps': 450_000,
    'learning_rate': 1e-02,
}


def xy_to_idx(key, width, height):
    shape = np.array(key).shape
    if len(shape) == 1:
        return key[0] + key[1]*width
    elif len(shape) == 2:
        return key[:,0] + key[:,1]*width
    else:
        raise NotImplementedError()


if __name__ == "__main__":

    env = env_suite.random_grid_env(**env_args, smooth_obs=False, one_hot_obs=False,
                                    distance_reward=False, absorb=False)

    print('Env num states:', env.num_states)
    print('Env num actions:', env.num_actions)
    print('Transition matrix shape:', env.transition_matrix().shape)
    print('Initial state distribution:', env.initial_state_distribution)
    print('Env render:')
    env.reset()
    env.render()
    print('\n')

    if args['algo'] == 'val_iter':
        val_iter_args['gamma'] = args['gamma']
        agent = value_iteration.ValueIteration(env, **val_iter_args)
    elif args['algo'] == 'q_learning':
        q_learning_args['gamma'] = args['gamma']
        agent = q_learning.QLearning(env, **q_learning_args)
    elif args['algo'] == 'dqn':
        dqn_args['discount'] = args['gamma']
        agent = dqn.DQN(env, dqn_args)
    else:
        raise ValueError("Unknown algorithm.")

    # Train agent.
    Q_vals, max_Q_vals, policy = agent.train(num_episodes=args['num_episodes'])

    #print('\nQ-vals:', Q_vals)
    print('Max Q-vals:', max_Q_vals)
    print('Policy:', policy)
    env.render()

    # Plot policy.
    print('Policy:')
    sys.stdout.write('-'*(env_args['size_x']+2)+'\n')
    for h in range(env_args['size_y']):
        sys.stdout.write('|')
        for w in range(env_args['size_x']):
            sys.stdout.write(str(policy[xy_to_idx((w,h), env_args['size_x'], env_args['size_y'])]))
        sys.stdout.write('|\n')
    sys.stdout.write('-' * (env_args['size_x'] + 2)+'\n')

    # Plot max Q-values.
    print('Max Q-vals:')
    for h in range(env_args['size_y']):
        for w in range(env_args['size_x']):
            sys.stdout.write("{:.1f} ".format(max_Q_vals[xy_to_idx((w,h), env_args['size_x'], env_args['size_y'])]))
        sys.stdout.write('\n')
    sys.stdout.write('\n')
