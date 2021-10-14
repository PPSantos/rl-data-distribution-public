import numpy as np
from tqdm import tqdm

from algos.utils.array_functions import choice_eps_greedy


class QLearning(object):

    def __init__(self, env, alpha, gamma,
            expl_eps_init, expl_eps_final, expl_eps_episodes, num_episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.expl_eps_init = expl_eps_init
        self.expl_eps_final = expl_eps_final
        self.expl_eps_episodes = expl_eps_episodes
        self.num_episodes = num_episodes

    def train(self, q_vals_period, replay_buffer_counts_period,
                num_rollouts, rollouts_period, rollouts_envs):

        Q = np.zeros((self.env.num_states, self.env.num_actions))

        episode_rewards = []
        states_counts = np.zeros((self.env.num_states))
        Q_vals = np.zeros((self.num_episodes, self.env.num_states, self.env.num_actions))

        for episode in tqdm(range(self.num_episodes)):

            s_t = self.env.reset()
            s_t = self.env.get_state()

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
                s_t1 = self.env.get_state()

                # Q-learning update.
                # if not done:
                Q[s_t][a_t] += self.alpha * \
                        (r_t1 + self.gamma * np.max(Q[s_t1,:]) - Q[s_t][a_t])
                # else:
                #    Q[s_t][a_t] += self.alpha * \
                #        (r_t1 + 0.0 - Q[s_t][a_t]) 

                # Log data.
                episode_cumulative_reward += r_t1
                states_counts[s_t] += 1

                s_t = s_t1

            episode_rewards.append(episode_cumulative_reward)
            Q_vals[episode,:,:] = Q

        data = {}
        data['episode_rewards'] = episode_rewards
        #data['states_counts'] = states_counts
        #data['Q_vals'] = Q_vals
        #data['policy'] = np.argmax(Q_vals[-1], axis=1)
        #data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)

        return data