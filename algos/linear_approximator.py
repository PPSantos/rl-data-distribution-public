import random
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


class LinearApproximator(object):

    def __init__(self, env, env_grid_spec, alpha, gamma,
            expl_eps_init, expl_eps_final, expl_eps_episodes,
            synthetic_replay_buffer, synthetic_replay_buffer_alpha,
            replay_size, batch_size):

        np.random.seed()
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.expl_eps_init = expl_eps_init
        self.expl_eps_final = expl_eps_final
        self.expl_eps_episodes = expl_eps_episodes
        self.synthetic_replay_buffer = synthetic_replay_buffer

        # Custom 'mdp_1' env. features.
        self.num_features = 5
        self.feature_map = np.array([
                        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], # state 0.
                        [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]], # state 1.
                        [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]], # state 2.
                        [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]]  # state 3.
        ], dtype=np.float32)

        self.weights = np.zeros(self.num_features)

        self.replay = ReplayBuffer(size=replay_size, statistics_table_shape=(self.env.num_states,self.env.num_actions))
        self.batch_size = batch_size
        self.replay_size = replay_size

        self.sampling_dist_size = self.env.num_states * self.env.num_actions
        self.sampling_dist = np.random.dirichlet([synthetic_replay_buffer_alpha]*self.sampling_dist_size)

        self.env_grid_spec = env_grid_spec

    def train(self, num_episodes, q_vals_period, replay_buffer_counts_period,
            num_rollouts, rollouts_period, phi, rollouts_phi):

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

        for episode in tqdm(range(num_episodes)):

            s_t = self.env.reset()

            # Calculate exploration epsilon.
            fraction = np.minimum(episode / self.expl_eps_episodes, 1.0)
            epsilon = self.expl_eps_init + fraction * (self.expl_eps_final - self.expl_eps_init)

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
                    self._update()

                # Log data.
                episode_cumulative_reward += r_t1
                states_counts[s_t] += 1
                steps_counter += 1

                s_t = s_t1

            episode_rewards.append(episode_cumulative_reward)

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
                print('Executing evaluation rollouts...')

                if self.env_grid_spec:
                    self.base_env.set_phi(rollouts_phi)

                r_rewards = []
                for i in range(num_rollouts):
                    r_rewards.append(self._execute_rollout())

                rollouts_episodes.append(episode)
                rollouts_rewards.append(r_rewards)

                if self.env_grid_spec:
                    self.base_env.set_phi(phi)

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

        return data

    def _update(self):

        # Sample from replay buffer.
        states, actions, rewards, next_states, _ = self.replay.sample(self.batch_size)

        for i in range(self.batch_size):
            s_t, a_t, r_t1, s_t1 = states[i], actions[i], rewards[i], next_states[i]

            q_vals_pred_next_state = [np.dot(self.weights, self.feature_map[s_t1,a]) for a in range(self.env.num_actions)]
            delta = (np.dot(self.weights, self.feature_map[s_t,a_t]) - (r_t1 + self.gamma*np.max(q_vals_pred_next_state))) \
                    * self.feature_map[s_t,a_t]
            self.weights = self.weights - self.alpha * delta

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

    def _execute_rollout(self):

        s_t = self.env.reset()

        episode_cumulative_reward = 0
        done = False
        while not done:

            # Pick action.
            q_vals_pred = [np.dot(self.weights, self.feature_map[s_t,a]) for a in range(self.env.num_actions)]
            a_t = choice_eps_greedy(q_vals_pred, epsilon=0.0)

            # Step.
            s_t1, r_t1, done, info = self.env.step(a_t)

            episode_cumulative_reward += r_t1

            s_t = s_t1

        return episode_cumulative_reward


class ReplayBuffer(object):
    def __init__(self, size, statistics_table_shape):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        self.statistics_table = np.zeros((statistics_table_shape))

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        self.statistics_table[obs_t,action] += 1

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            _next_id_data = self._storage[self._next_idx]
            (next_id_obs_t, next_id_action, next_id_reward, next_id_obs_tp1, next_id_done) = _next_id_data
            self.statistics_table[next_id_obs_t,next_id_action] -= 1
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
    def get_replay_buffer_counts(self):
        return self.statistics_table.copy()