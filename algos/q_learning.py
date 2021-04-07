import sys
import numpy as np


def all_eq(values):
    # Returns True if every element of 'values' is the same.
    return all(np.isnan(values)) or max(values) - min(values) < 1e-6

def choice_eps_greedy(values, epsilon):
    if np.random.rand() <= epsilon or all_eq(values):
        #print('explored')
        return np.random.choice(len(values))
    else:
        #print('exploited')
        return np.argmax(values)


class QLearning(object):

    def __init__(self, env, alpha, gamma, expl_eps):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.expl_eps = expl_eps

    def train(self, num_episodes):

        Q = np.zeros((self.env.num_states, self.env.num_actions))

        for episode in range(num_episodes):
            # print(f"Episode {episode}:")

            Q_old = np.copy(Q)

            s_t = self.env.reset()
            #print('s_t:', s_t)

            done = False
            while not done:

                # Pick action.
                a_t = choice_eps_greedy(Q[s_t], self.expl_eps)
                #print('action:', a_t)

                # Env step.
                s_t1, r_t1, done, info = self.env.step(a_t)
                #print('r_t1', r_t1)
                #print('s_t1', s_t1)
                #self.env.render()

                # Q-learning update.
                Q[s_t][a_t] += self.alpha * \
                    (r_t1 + self.gamma * np.max(Q[s_t1,:]) - Q[s_t][a_t])

                s_t = s_t1

            delta = np.sum(np.abs(Q - Q_old))
            print('Delta:', delta)

        # Calculate policy.
        policy = {}
        max_Q_vals = {}
        for state in range(self.env.num_states):
            policy[state] = np.argmax(Q[state])
            max_Q_vals[state] = np.max(Q[state])

        self.Q = Q
        self.max_Q_vals = max_Q_vals
        self.policy = policy

        return Q, max_Q_vals, policy