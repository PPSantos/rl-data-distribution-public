import json
from typing import Sequence

from tqdm import tqdm
import numpy as np

import sonnet as snt

import dm_env

import acme
from acme import wrappers
from acme.utils import loggers

from algos.dqn import dqn_acme
from rlutil.envs.gridcraft.grid_spec_cy import TileType


def _make_network(env_spec : dm_env,
                  hidden_layers : Sequence[int] = [10,10]):
    layers = hidden_layers + [env_spec.actions.num_values]
    print('network layers:', layers)
    network = snt.Sequential([
        snt.nets.MLP(layers, activate_final=False),
    ])
    return network

def wrap_env(env):
    return wrappers.wrap_all(env, [
        wrappers.GymWrapper,
        wrappers.SinglePrecisionWrapper,
    ])


class DQN(object):

    def __init__(self, env, env_grid_spec, log_path, dqn_args):

        np.random.seed()

        self.base_env = env
        self.env_grid_spec = env_grid_spec
        self.env = wrap_env(env)
        env_spec = acme.make_environment_spec(self.env)

        network = _make_network(env_spec,
                        hidden_layers=dqn_args['hidden_layers'])

        self.agent = dqn_acme.DQN(environment_spec=env_spec,
                                    network=network,
                                    batch_size=dqn_args['batch_size'],
                                    target_update_period=dqn_args['target_update_period'],
                                    samples_per_insert=dqn_args['samples_per_insert'],
                                    min_replay_size=dqn_args['min_replay_size'],
                                    max_replay_size=dqn_args['max_replay_size'],
                                    epsilon_init=dqn_args['epsilon_init'],
                                    epsilon_final=dqn_args['epsilon_final'],
                                    epsilon_schedule_timesteps=dqn_args['epsilon_schedule_timesteps'],
                                    learning_rate=dqn_args['learning_rate'],
                                    discount=dqn_args['discount'],
                                    synthetic_replay_buffer=dqn_args['synthetic_replay_buffer'],
                                    num_states=self.base_env.num_states,
                                    num_actions=self.base_env.num_actions,
                                    logger=loggers.CSVLogger(directory_or_file=log_path, label='learner'))

        # Internalise arguments.
        self.oracle_q_vals = dqn_args['oracle_q_vals']
        self.discount = dqn_args['discount']
        self.batch_size = dqn_args['batch_size']

        self.synthetic_replay_buffer = dqn_args['synthetic_replay_buffer']
        if self.synthetic_replay_buffer:
            self.synthetic_static_dataset_size = 500 # in episodes.
            self.sampling_dist_size = self.base_env.num_states * self.base_env.num_actions
            if dqn_args['custom_sampling_dist'] is None:
                self.sampling_dist = np.random.dirichlet([dqn_args['synthetic_replay_buffer_alpha']]*self.sampling_dist_size)
            else:
                # Load custom sampling dist.
                custom_sampling_dist_path = dqn_args['custom_sampling_dist']
                print(f'Loading custom sampling dist from {custom_sampling_dist_path}')    
                with open(custom_sampling_dist_path, 'r') as f:
                    sampling_dist_data = json.load(f)
                    sampling_dist_data = json.loads(sampling_dist_data)
                f.close()
                self.sampling_dist = sampling_dist_data['stationary_dist']

            print('self.sampling_dist (synthetic replay buffer dataset):', self.sampling_dist)
            print('self.sampling_dist_size (S*A):', self.sampling_dist_size)

    def train(self, num_episodes, q_vals_period, replay_buffer_counts_period,
                num_rollouts, rollouts_period, rollouts_envs, compute_e_vals):

        rollouts_envs = [wrap_env(e) for e in rollouts_envs]

        states_counts = np.zeros((self.env.num_states))
        episode_rewards = []

        Q_vals = np.zeros((num_episodes//q_vals_period,
                self.base_env.num_states, self.base_env.num_actions))
        Q_vals_episodes = []
        Q_vals_ep = 0

        if compute_e_vals:
            E_vals = np.zeros((num_episodes//q_vals_period,
                    self.base_env.num_states, self.base_env.num_actions))
            Q_errors = np.zeros((num_episodes//q_vals_period,
                    self.base_env.num_states, self.base_env.num_actions))

        replay_buffer_counts_episodes = []
        replay_buffer_counts = []

        rollouts_episodes = []
        rollouts_rewards = []

        for episode in tqdm(range(num_episodes)):

            if self.synthetic_replay_buffer and \
                (episode % self.synthetic_static_dataset_size == 0):
                # Create dataset with size = self.synthetic_static_dataset_size*self.base_env.time_limit
                static_dataset = self._create_dataset()
                static_dataset_iterator = iter(static_dataset)

            timestep = self.env.reset()
            env_state = self.base_env.get_state()

            self.agent.observe_first(timestep)

            episode_cumulative_reward = 0
            while not timestep.last():

                action = self.agent.select_action(timestep.observation)
                timestep = self.env.step(action)

                self.agent.observe_with_extras(action,
                    next_timestep=timestep, extras=(np.int32(env_state), np.int32(self.base_env.get_state())))

                if self.synthetic_replay_buffer:
                    # Insert transition from the static dataset.
                    transition, extras = next(static_dataset_iterator)
                    self.agent.add_to_replay_buffer(transition, extras)

                self.agent.update()

                env_state = self.base_env.get_state()

                # Log data.
                episode_cumulative_reward += timestep.reward
                states_counts[self.base_env.get_state()] += 1

            episode_rewards.append(episode_cumulative_reward)

            # Store current Q-values (and E-values).
            if episode % q_vals_period == 0:
                print('Storing current Q-values estimates.')
                estimated_Q_vals = np.zeros((self.env.num_states, self.env.num_actions))
                for state in range(self.base_env.num_states):
                    if self.env_grid_spec:
                        xy = self.env_grid_spec.idx_to_xy(state)
                        tile_type = self.env_grid_spec.get_value(xy)
                        if tile_type == TileType.WALL:
                            estimated_Q_vals[state,:] = 0
                        else:
                            obs = self.base_env.observation(state)
                            qvs = self.agent.get_Q_vals(obs)
                            estimated_Q_vals[state,:] = qvs
                    else:
                        obs = self.base_env.observation(state)
                        qvs = self.agent.get_Q_vals(obs)
                        estimated_Q_vals[state,:] = qvs

                Q_vals_episodes.append(episode)
                Q_vals[Q_vals_ep,:,:] = estimated_Q_vals

                # Estimate E-values for the current set of Q-values.
                if compute_e_vals:
                    print('Estimating E-values for the current set of Q-values.')
                    _E_vals = np.zeros((self.env.num_states, self.env.num_actions))
                    _q_errors =  np.zeros((self.env.num_states, self.env.num_actions))
                    _samples_counts = np.zeros((self.env.num_states, self.env.num_actions))

                    for _ in range(10_000):

                        # Sample from replay buffer.
                        data = self.agent.sample_replay_buffer_batch()
                        _, actions, rewards, _, _, states, next_states = data

                        for i in range(self.batch_size):
                            s_t, a_t, r_t1, s_t1 = states[i], actions[i], rewards[i], next_states[i]
                            # e_t1 = np.abs(estimated_Q_vals[s_t, a_t] - self.oracle_q_vals[s_t, a_t]) # oracle target
                            e_t1 = np.abs(estimated_Q_vals[s_t, a_t] - (r_t1 + self.discount*np.max(estimated_Q_vals[s_t1,:]))) # TD target.

                            _E_vals[s_t][a_t] += 0.05 * \
                            (e_t1 + self.discount * np.max(_E_vals[s_t1,:]) - _E_vals[s_t][a_t])

                            _samples_counts[s_t,a_t] += 1
                            _q_errors[s_t,a_t] += e_t1

                    E_vals[Q_vals_ep,:,:] = _E_vals
                    Q_errors[Q_vals_ep,:,:] = _q_errors / (_samples_counts + 1e-05)

                Q_vals_ep += 1

            # Get replay buffer statistics.
            if (episode > 1) and (episode % replay_buffer_counts_period == 0):
                print('Getting replay buffer statistics.')
                replay_buffer_counts_episodes.append(episode)
                replay_buffer_counts.append(self.agent.get_replay_buffer_counts())

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
        data['Q_vals_episodes'] = Q_vals_episodes
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)
        data['policy'] = np.argmax(Q_vals[-1], axis=1)
        data['replay_buffer_counts_episodes'] = replay_buffer_counts_episodes
        data['replay_buffer_counts'] = replay_buffer_counts
        data['rollouts_episodes'] = rollouts_episodes
        data['rollouts_rewards'] = rollouts_rewards
        if compute_e_vals:
            data['E_vals'] = E_vals
            data['Q_errors'] = Q_errors

        return data

    def _create_dataset(self):
        print('Creating static dataset of transitions...')

        static_dataset = []
        dataset_size = self.synthetic_static_dataset_size * self.base_env.time_limit
        mesh = np.array(np.meshgrid(np.arange(self.base_env.num_states),
                                    np.arange(self.base_env.num_actions)))
        sa_combinations = mesh.T.reshape(-1, 2)

        sa_counts = np.zeros((self.base_env.num_states, self.base_env.num_actions))

        for _ in range(dataset_size):

            # Randomly sample (state, action) pair.
            if self.env_grid_spec:
                tile_type = TileType.WALL
                while tile_type == TileType.WALL:
                    sampled_idx = np.random.choice(np.arange(self.sampling_dist_size), p=self.sampling_dist)
                    state, action = sa_combinations[sampled_idx]
                    xy = self.env_grid_spec.idx_to_xy(state)
                    tile_type = self.env_grid_spec.get_value(xy)
            else:
                sampled_idx = np.random.choice(np.arange(self.sampling_dist_size), p=self.sampling_dist)
                state, action = sa_combinations[sampled_idx]

            sa_counts[state, action] += 1

            observation = self.base_env.observation(state)

            # Sample next state, observation and reward.
            self.base_env.set_state(state)
            next_observation, reward, done, info = self.base_env.step(action)

            transition = (observation, np.array(action, dtype=np.int32),
                            np.array(reward, dtype=np.float32),
                            np.array(1.0, dtype=np.float32),
                            next_observation)
            extras = (np.int32(state), np.int32(self.base_env.get_state()))

            static_dataset.append((transition, extras))

            self.base_env.reset()

        # Correct dataset in order to ensure coverage over all (state, action) pairs.
        # (this corection barely changes the data dist. entropy).
        zero_positions = np.where(sa_counts == 0)
        for (state,action) in zip(*zero_positions):

            observation = self.base_env.observation(state)

            # Sample next state, observation and reward.
            self.base_env.set_state(state)
            next_observation, reward, done, info = self.base_env.step(action)

            transition = (observation, np.array(action, dtype=np.int32),
                            np.array(reward, dtype=np.float32),
                            np.array(1.0, dtype=np.float32),
                            next_observation)
            extras = (np.int32(state), np.int32(self.base_env.get_state()))

            static_dataset.append((transition, extras))

        print(f'Static dataset created containing {len(static_dataset)} transitions.')

        return static_dataset

    def _execute_rollout(self, r_env):

        timestep = r_env.reset()

        rollout_cumulative_reward = 0
        while not timestep.last():
            action = self.agent.deterministic_action(timestep.observation)
            timestep = r_env.step(action)
            rollout_cumulative_reward += timestep.reward

        return rollout_cumulative_reward
