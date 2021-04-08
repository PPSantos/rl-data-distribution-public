import numpy as np
from tqdm import tqdm

def all_eq(values):
    # Returns True if every element of 'values' is the same.
    return all(np.isnan(values)) or max(values) - min(values) < 1e-6

def choice_eps_greedy(values, epsilon):
    if np.random.rand() <= epsilon or all_eq(values):
        return np.random.choice(len(values))
    else:
        return np.argmax(values)


class QLearning(object):

    def __init__(self, env, alpha, gamma,
            expl_eps_init, expl_eps_final, expl_eps_episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.expl_eps_init = expl_eps_init
        self.expl_eps_final = expl_eps_final
        self.expl_eps_episodes = expl_eps_episodes

    def train(self, num_episodes):

        Q = np.zeros((self.env.num_states, self.env.num_actions))
        # counts = np.zeros((self.env.num_states))

        for episode in tqdm(range(num_episodes)):

            s_t = self.env.reset()

            # Calculate exploration epsilon.
            fraction = np.minimum(episode / self.expl_eps_episodes, 1.0)
            epsilon = self.expl_eps_init + fraction * (self.expl_eps_final - self.expl_eps_init)

            done = False
            while not done:

                # counts[s_t] += 1

                # Pick action.
                a_t = choice_eps_greedy(Q[s_t], epsilon)

                # Env step.
                s_t1, r_t1, done, info = self.env.step(a_t)

                # Q-learning update.
                Q[s_t][a_t] += self.alpha * \
                    (r_t1 + self.gamma * np.max(Q[s_t1,:]) - Q[s_t][a_t])

                s_t = s_t1

        # print(counts)

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