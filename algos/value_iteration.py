import numpy as np


class ValueIteration(object):

    def __init__(self, env, gamma, epsilon):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    def compute(self):

        # Construct reward function and transition probability function.
        transition_function = {}
        reward_function = {}
        for state in range(self.env.num_states):

            aux_transition_f = {}
            aux_reward_f = {}
            for action in range(self.env.num_actions):
                aux_transition_f[action] = self.env.get_transitions(state, action) # dict of the form {next_state: prob}
                aux_reward_f[action] = self.env.get_reward(state, action)
            
            transition_function[state] = aux_transition_f
            reward_function[state] = aux_reward_f

        print('Running value iteration...')
        Q_vals = np.zeros((self.env.num_states, self.env.num_actions))
        while True:
            Q_vals_old = np.copy(Q_vals)

            for state in range(self.env.num_states):
                for action in range(self.env.num_actions):

                    Q_next_states = np.sum([p*np.max(Q_vals[next_state,:])
                                    for (next_state,p) in transition_function[state][action].items()])

                    Q_vals[state][action] = reward_function[state][action] + \
                        self.gamma * Q_next_states

            delta = np.sum(np.abs(Q_vals - Q_vals_old))
            print('Delta:', delta)

            if delta < self.epsilon:
                break

        data = {}
        data['Q_vals'] = Q_vals

        # print('Value iteration solution:')
        # print(Q_vals)

        return data
