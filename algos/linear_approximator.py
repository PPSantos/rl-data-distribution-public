import numpy as np
from tqdm import tqdm

from utils.array_functions import choice_eps_greedy
from algos.numpy_replay_buffer import NumpyReplayBuffer

class LinearApproximator(object):

    def __init__(self, env, env_grid_spec, alpha_init, alpha_final,
                alpha_schedule_episodes, gamma,
                expl_eps_init, expl_eps_final, expl_eps_episodes,
                synthetic_replay_buffer, synthetic_replay_buffer_alpha,
                replay_size, batch_size):

        np.random.seed()
        
        self.env = env
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.alpha_schedule_episodes = alpha_schedule_episodes
        self.gamma = gamma
        self.expl_eps_init = expl_eps_init
        self.expl_eps_final = expl_eps_final
        self.expl_eps_episodes = expl_eps_episodes
        self.synthetic_replay_buffer = synthetic_replay_buffer

        # Custom 'mdp_1' env. features.
        self.num_features = 5
        self.feature_map = np.array([
                        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], # state 0.
                        [[1.2, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]], # state 1.
                        [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]], # state 2.
                        [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]]  # state 3.
        ], dtype=np.float32)

        self.weights = np.zeros(self.num_features)

        self.replay = NumpyReplayBuffer(size=replay_size, statistics_table_shape=(self.env.num_states,self.env.num_actions))
        self.batch_size = batch_size
        self.replay_size = replay_size

        self.sampling_dist_size = self.env.num_states * self.env.num_actions
        self.sampling_dist = np.random.dirichlet([synthetic_replay_buffer_alpha]*self.sampling_dist_size)

        self.env_grid_spec = env_grid_spec

    def train(self, num_episodes, q_vals_period, replay_buffer_counts_period,
            num_rollouts, rollouts_period, rollouts_envs, compute_e_vals):

        if self.synthetic_replay_buffer:
            self._prefill_replay_buffer()

        episode_rewards = []
        states_counts = np.zeros((self.env.num_states))

        Q_vals = np.zeros((num_episodes//q_vals_period,
                self.env.num_states, self.env.num_actions))
        Q_vals_episodes = []

        replay_buffer_counts_episodes = []
        replay_buffer_counts = []

        rollouts_episodes = []
        rollouts_rewards = []

        steps_counter = 0
        Q_vals_ep = 0

        weights_list = []

        for episode in tqdm(range(num_episodes)):

            s_t = self.env.reset()

            # Calculate exploration epsilon.
            fraction = np.minimum(episode / self.expl_eps_episodes, 1.0)
            epsilon = self.expl_eps_init + fraction * (self.expl_eps_final - self.expl_eps_init)

            # Calculate learning rate (alpha).
            fraction = np.minimum(episode / self.alpha_schedule_episodes, 1.0)
            alpha = self.alpha_init + fraction * (self.alpha_final - self.alpha_init)     

            done = False
            episode_cumulative_reward = 0
            while not done:

                # Pick action.
                q_vals_pred = [np.dot(self.weights, self.feature_map[s_t,a]) for a in range(self.env.num_actions)]
                a_t = choice_eps_greedy(q_vals_pred, epsilon)

                # Env step.
                s_t1, r_t1, done, info = self.env.step(a_t)

                # Add to replay buffer.
                if not self.synthetic_replay_buffer:
                    self.replay.add(s_t, a_t, r_t1, s_t1, done)

                # Weights update.
                if steps_counter > self.batch_size: # and (steps_counter % self.batch_size) == 0:
                    self._update(alpha)

                # Log data.
                episode_cumulative_reward += r_t1
                states_counts[s_t] += 1
                steps_counter += 1

                s_t = s_t1

            episode_rewards.append(episode_cumulative_reward)

            weights_list.append(self.weights)

            # Get Q-values.
            if episode % q_vals_period == 0:
                Q_vals_episodes.append(episode)
                for s in range(self.env.num_states):
                    for a in range(self.env.num_actions):
                        #print(s, a, np.dot(self.weights, self.feature_map[s,a]))
                        Q_vals[Q_vals_ep,s,a] = np.dot(self.weights, self.feature_map[s,a])
                Q_vals_ep += 1

            # Estimate statistics of the replay buffer contents.
            if (episode > 1) and (episode % replay_buffer_counts_period == 0):
                print('Appending replay buffer info...')
                replay_buffer_counts_episodes.append(episode)
                replay_buffer_counts.append(self.replay.get_replay_buffer_counts())

            # Execute evaluation rollouts.
            if episode % rollouts_period == 0:
                print('Executing evaluation rollouts.')

                r_rewards = []
                for r_env in rollouts_envs:
                    r_rewards.append([self._execute_rollout(r_env) for _ in range(num_rollouts)])

                rollouts_episodes.append(episode)
                rollouts_rewards.append(r_rewards)

        data = {}
        data['episode_rewards'] = episode_rewards
        data['states_counts'] = states_counts
        data['Q_vals'] = Q_vals
        data['Q_vals_episodes'] = Q_vals_episodes
        data['policy'] = np.argmax(Q_vals[-1], axis=1)
        data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)
        data['replay_buffer_counts_episodes'] = replay_buffer_counts_episodes
        data['replay_buffer_counts'] = replay_buffer_counts
        data['rollouts_episodes'] = rollouts_episodes
        data['rollouts_rewards'] = rollouts_rewards
        data['weights'] = weights_list

        return data

    def _update(self, alpha):

        # Sample from replay buffer.
        states, actions, rewards, next_states, _ = self.replay.sample(self.batch_size)

        for i in range(self.batch_size):
            s_t, a_t, r_t1, s_t1 = states[i], actions[i], rewards[i], next_states[i]

            q_vals_pred_next_state = [np.dot(self.weights, self.feature_map[s_t1,a]) for a in range(self.env.num_actions)]
            delta = (np.dot(self.weights, self.feature_map[s_t,a_t]) - (r_t1 + self.gamma*np.max(q_vals_pred_next_state))) \
                    * self.feature_map[s_t,a_t]
            self.weights = self.weights - alpha * delta

    def _prefill_replay_buffer(self):
        print('Prefilling replay buffer...')

        mesh = np.array(np.meshgrid(np.arange(self.env.num_states),
                                    np.arange(self.env.num_actions)))
        sa_combinations = mesh.T.reshape(-1, 2)

        for _ in range(self.replay_size):

            # Randomly sample (state, action) pair.
            sampled_idx = np.random.choice(np.arange(self.sampling_dist_size), p=self.sampling_dist)
            state, action = sa_combinations[sampled_idx]

            # Sample next state, observation and reward.
            self.env.set_state(state)
            next_state, reward, done, info = self.env.step(action)

            # Insert to replay buffer.
            self.replay.add(state, action, reward, next_state, False)

            self.env.reset()

    def _execute_rollout(self, r_env):

        s_t = r_env.reset()

        episode_cumulative_reward = 0
        done = False
        while not done:

            # Pick action.
            q_vals_pred = [np.dot(self.weights, self.feature_map[s_t,a]) for a in range(r_env.num_actions)]
            a_t = choice_eps_greedy(q_vals_pred, epsilon=0.0)

            # Step.
            s_t1, r_t1, done, info = r_env.step(a_t)

            episode_cumulative_reward += r_t1

            s_t = s_t1

        return episode_cumulative_reward
