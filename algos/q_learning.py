import numpy as np
from tqdm import tqdm

from utils.array_functions import choice_eps_greedy
from algos.numpy_replay_buffer import NumpyReplayBuffer

class QLearning(object):

    def __init__(self, env, qlearning_args):
        self.env = env
        self.alpha_init = qlearning_args['alpha_init']
        self.alpha_final = qlearning_args['alpha_final']
        self.alpha_steps = qlearning_args['alpha_steps']
        self.gamma = qlearning_args['gamma']

        self.replay_buffer = NumpyReplayBuffer(size=qlearning_args['replay_buffer_size'])
        self.batch_size = qlearning_args['replay_buffer_batch_size']

    def train_offline(self, learning_steps):

        # Prefill replay buffer.
        print('Pre-filling replay buffer.')
        for _ in range(20):
            for state in range(self.env.num_states):
                for action in range(self.env.num_actions):
                    self.env.reset()
                    self.env.set_state(state)
                    s_t1, r_t1, done, info = self.env.step(action)
                    s_t1 = self.env.get_state()
                    self.replay_buffer.add(state, action, r_t1, s_t1, done, info)

        Q = np.zeros((self.env.num_states, self.env.num_actions))
        Q_old = np.copy(Q)

        for step in tqdm(range(learning_steps)):

            # Calculate learning rate.
            fraction = np.minimum(step / self.alpha_steps, 1.0)
            alpha = self.alpha_init + fraction * (self.alpha_final - self.alpha_init)

            # Update.
            states, actions, rewards, next_states, dones, infos = self.replay_buffer.sample(self.batch_size)

            for i in range(self.batch_size):
                state, action, reward, next_state, done, info = \
                    states[i], actions[i], rewards[i], next_states[i], dones[i], infos[i]

                is_truncated = info.get('TimeLimit.truncated', False)
                # Q-learning update.
                if done and (not is_truncated):
                    Q[state][action] += alpha * (reward - Q[state][action])
                else:
                    Q[state][action] += alpha * \
                            (reward + self.gamma * np.max(Q[next_state,:]) - Q[state][action])

            if step % 1_000 == 0:
                print('Alpha:', alpha)
                print('Q tab error:', np.sum(np.abs(Q-Q_old)))
                self._execute_rollout(Q)
                Q_old = np.copy(Q)

        data = {}
        data['Q_vals'] = Q

        return data

    def _execute_rollout(self, Q_vals):

        s_t = self.env.reset()
        s_t = self.env.get_state()
        terminated = False
        episode_cumulative_reward = 0

        while not terminated:

            # Pick action.
            a_t = choice_eps_greedy(Q_vals[s_t], epsilon=0.0)

            # Env step.
            s_t1, r_t1, terminated, info = self.env.step(a_t)
            s_t1 = self.env.get_state()

            episode_cumulative_reward += r_t1

            s_t = s_t1

        print('Rollout episode reward:', episode_cumulative_reward)
