import numpy as np


class ValueIteration(object):

    def __init__(self, env, gamma, epsilon):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    def train(self, num_episodes):

        Q_vals = np.zeros((self.env.num_states, self.env.num_actions))

        print('Running value iteration...')
        while True:
            Q_vals_old = np.copy(Q_vals)
            
            for state in range(self.env.num_states):
                for action in range(self.env.num_actions):

                    Q_vals[state][action] = self.env.reward(state, 0, 0) + \
                        self.gamma * np.dot(self.env.transition_matrix()[state][action], \
                        np.max(Q_vals, axis=1))

            delta = np.sum(np.abs(Q_vals - Q_vals_old))
            print('Delta:', delta)

            if delta < self.epsilon:
                break

        # Calculate policy.
        policy = {}
        max_Q_vals = {}
        for state in range(self.env.num_states):
            policy[state] = np.argmax(Q_vals[state])
            max_Q_vals[state] = np.max(Q_vals[state])

        data = {}
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = max_Q_vals
        data['policy'] = policy

        return data
