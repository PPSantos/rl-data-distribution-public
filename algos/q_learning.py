import numpy as np
from tqdm import tqdm

from utils.array_functions import choice_eps_greedy
from algos.numpy_replay_buffer import NumpyReplayBuffer

class QLearning(object):

    def __init__(self, env, qlearning_args):
        self.env = env
        self.alpha = qlearning_args['alpha']
        self.gamma = qlearning_args['gamma']
        self.expl_eps_init = qlearning_args['expl_eps_init']
        self.expl_eps_final = qlearning_args['expl_eps_final']
        self.expl_eps_episodes = qlearning_args['expl_eps_episodes']
        self.num_episodes = qlearning_args['num_episodes']

        self.replay_buffer = NumpyReplayBuffer(size=qlearning_args['replay_buffer_size'],
                                                statistics_table_shape=(self.env.num_states,self.env.num_actions))
        self.batch_size = qlearning_args['replay_buffer_batch_size']

    def train_offline(self, learning_steps): 
        # Prefill replay buffer.
        print('Pre-filling replay buffer.')
        for _ in range(10):
            for state in range(self.env.num_states):
                for action in range(self.env.num_actions):
                    self.env.reset()
                    self.env.set_state(state)
                    s_t1, r_t1, done, _ = self.env.step(action)
                    s_t1 = self.env.get_state()
                    self.replay_buffer.add(state, action, r_t1, s_t1, False)

        Q = np.zeros((self.env.num_states, self.env.num_actions))

        for _ in tqdm(range(learning_steps)):

            # Update.
            states, actions, rewards, next_states, _ = self.replay_buffer.sample(self.batch_size)

            for i in range(self.batch_size):
                state, action, reward, next_state = states[i], actions[i], rewards[i], next_states[i]

                # Q-learning update.
                Q[state][action] += self.alpha * \
                        (reward + self.gamma * np.max(Q[next_state,:]) - Q[state][action])

        data = {}
        data['Q_vals'] = Q
        return data


    def train_online(self):

        # Prefill replay buffer.
        print('Pre-filling replay buffer.')
        for _ in range(10):
            for state in range(self.env.num_states):
                for action in range(self.env.num_actions):
                    self.env.reset()
                    self.env.set_state(state)
                    s_t1, r_t1, done, _ = self.env.step(action)
                    s_t1 = self.env.get_state()
                    self.replay_buffer.add(state, action, r_t1, s_t1, False)

        Q = np.zeros((self.env.num_states, self.env.num_actions))

        episode_rewards = []
        for episode in tqdm(range(self.num_episodes)):

            s_t = self.env.reset()
            s_t = self.env.get_state()

            # Calculate exploration epsilon.
            fraction = np.minimum(episode / self.expl_eps_episodes, 1.0)
            epsilon = self.expl_eps_init + fraction * (self.expl_eps_final - self.expl_eps_init)

            done = False
            episode_cumulative_reward = 0
            steps_counter = 0
            while not done:

                # Pick action.
                a_t = choice_eps_greedy(Q[s_t], epsilon)

                # Env step.
                s_t1, r_t1, done, info = self.env.step(a_t)
                s_t1 = self.env.get_state()

                # Add to replay buffer.
                self.replay_buffer.add(s_t, a_t, r_t1, s_t1, False)

                # Update.
                if steps_counter > self.batch_size:
                    states, actions, rewards, next_states, _ = self.replay_buffer.sample(self.batch_size)

                    for i in range(self.batch_size):
                        state, action, reward, next_state = states[i], actions[i], rewards[i], next_states[i]

                        # Q-learning update.
                        Q[state][action] += self.alpha * \
                                (reward + self.gamma * np.max(Q[next_state,:]) - Q[state][action])

                # Log data.
                episode_cumulative_reward += r_t1
                steps_counter += 1

                s_t = s_t1

            episode_rewards.append(episode_cumulative_reward)

        data = {}
        data['Q_vals'] = Q
        data['episode_rewards'] = episode_rewards

        return data
