import numpy as np
from tqdm import tqdm

from algos.utils.array_functions import choice_eps_greedy
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

    def train(self, q_vals_period, replay_buffer_counts_period,
                num_rollouts, rollouts_period, rollouts_envs):

        Q = np.zeros((self.env.num_states, self.env.num_actions))

        Q_vals = np.zeros((self.num_episodes//q_vals_period,
                self.env.num_states, self.env.num_actions))
        Q_vals_steps = []
        Q_vals_idx = 0

        rollouts_steps = []
        rollouts_rewards = []

        replay_buffer_counts_steps = []
        replay_buffer_counts = []

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

            # Store current Q-values.
            if episode % q_vals_period == 0:
                print('Storing current Q-values estimates.')
                Q_vals_steps.append(episode)
                Q_vals[Q_vals_idx,:,:] = Q
                Q_vals_idx += 1

            # Execute evaluation rollouts.
            if episode % rollouts_period == 0:
                print('Executing evaluation rollouts.')

                r_rewards = []
                for r_env in rollouts_envs:
                    r_rewards.append([self._execute_rollout(r_env, Q) for _ in range(num_rollouts)])

                rollouts_steps.append(episode)
                rollouts_rewards.append(r_rewards)

            # Get replay buffer statistics.
            if (episode > 1) and (episode % replay_buffer_counts_period == 0):
                print('Getting replay buffer statistics.')
                replay_buffer_counts_steps.append(episode)
                replay_buffer_counts.append(self.replay_buffer.get_replay_buffer_counts())

        data = {}
        data['Q_vals'] = Q_vals
        data['Q_vals_steps'] = Q_vals_steps
        data['rollouts_rewards'] = rollouts_rewards
        data['rollouts_steps'] = rollouts_steps
        data['replay_buffer_counts'] = replay_buffer_counts
        data['replay_buffer_counts_steps'] = replay_buffer_counts_steps

        return data

    def _execute_rollout(self, r_env, Q):
        s_t = r_env.reset()
        s_t = r_env.get_state()
        rollout_cumulative_reward = 0
        done = False
        while not done:
            a_t = choice_eps_greedy(Q[s_t,:], epsilon=0.0)
            s_t1, r_t1, done, _ = r_env.step(a_t)
            s_t1 = r_env.get_state()
            rollout_cumulative_reward += r_t1
            s_t = s_t1

        return rollout_cumulative_reward
