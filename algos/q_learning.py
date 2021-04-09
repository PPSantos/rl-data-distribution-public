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

        epsilon_values = []
        episode_rewards = []
        states_counts = np.zeros((self.env.num_states))
        Q_vals = np.zeros((num_episodes, self.env.num_states, self.env.num_actions))

        for episode in tqdm(range(num_episodes)):

            s_t = self.env.reset()

            # Calculate exploration epsilon.
            fraction = np.minimum(episode / self.expl_eps_episodes, 1.0)
            epsilon = self.expl_eps_init + fraction * (self.expl_eps_final - self.expl_eps_init)

            done = False
            episode_cumulative_reward = 0
            while not done:

                # Pick action.
                a_t = choice_eps_greedy(Q[s_t], epsilon)

                # Env step.
                s_t1, r_t1, done, info = self.env.step(a_t)

                # Q-learning update.
                Q[s_t][a_t] += self.alpha * \
                    (r_t1 + self.gamma * np.max(Q[s_t1,:]) - Q[s_t][a_t])

                # Log data.
                episode_cumulative_reward += r_t1
                states_counts[s_t] += 1

                s_t = s_t1

            epsilon_values.append(epsilon)
            episode_rewards.append(episode_cumulative_reward)
            Q_vals[episode,:,:] = Q

        # Calculate policy.
        policy = {}
        max_Q_vals = {}
        for state in range(self.env.num_states):
            policy[state] = np.argmax(Q[state])
            max_Q_vals[state] = np.max(Q[state])

        data = {}
        data['episode_rewards'] = episode_rewards
        data['epsilon_values'] = epsilon_values
        data['states_counts'] = states_counts
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = max_Q_vals
        data['policy'] = policy

        return data
