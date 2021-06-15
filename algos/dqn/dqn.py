import os
import random
from typing import Sequence

from tqdm import tqdm
import numpy as np

import dm_env

import tensorflow as tf
import sonnet as snt

import acme
from acme import specs
from acme.tf import networks
from acme import wrappers

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

    def __init__(self, env, env_grid_spec, dqn_args):

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
                                    prioritized_replay=dqn_args['prioritized_replay'],
                                    importance_sampling_exponent=dqn_args['importance_sampling_exponent'],
                                    priority_exponent=dqn_args['priority_exponent'],
                                    epsilon_init=dqn_args['epsilon_init'],
                                    epsilon_final=dqn_args['epsilon_final'],
                                    epsilon_schedule_timesteps=dqn_args['epsilon_schedule_timesteps'],
                                    learning_rate=dqn_args['learning_rate'],
                                    discount=dqn_args['discount'],
                                    synthetic_replay_buffer=dqn_args['synthetic_replay_buffer'],
                                    num_states=self.base_env.num_states,
                                    num_actions=self.base_env.num_actions)

        self.synthetic_replay_buffer = dqn_args['synthetic_replay_buffer']
        self.synthetic_static_dataset_size = 500 # in episodes.
        self.synthetic_replay_buffer_alpha = dqn_args['synthetic_replay_buffer_alpha']
        self.sampling_dist_size = self.base_env.num_states * self.base_env.num_actions
        self.sampling_dist = np.random.dirichlet([self.synthetic_replay_buffer_alpha]*self.sampling_dist_size)
        if self.synthetic_replay_buffer:
            print('self.sampling_dist (synthetic replay buffer dataset):', self.sampling_dist)
            print('self.sampling_dist_size:', self.sampling_dist_size)

    def train(self, num_episodes, q_vals_period, replay_buffer_counts_period,
            num_rollouts, rollouts_period, phi, rollouts_phi):

        states_counts = np.zeros((self.env.num_states))
        episode_rewards = []

        Q_vals = np.zeros((num_episodes//q_vals_period,
                self.base_env.num_states, self.base_env.num_actions))
        Q_vals_episodes = []
        Q_vals_ep = 0

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
                    next_timestep=timestep, extras=(np.int32(env_state),))

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

            # Store current Q-values.
            if episode % q_vals_period == 0:
                Q_vals_episodes.append(episode)
                for state in range(self.base_env.num_states):
                    if self.env_grid_spec:
                        xy = self.env_grid_spec.idx_to_xy(state)
                        tile_type = self.env_grid_spec.get_value(xy)
                        if tile_type == TileType.WALL:
                            Q_vals[Q_vals_ep,state,:] = 0
                        else:
                            obs = self.base_env.observation(state)
                            qvs = self.agent.get_Q_vals(obs)
                            Q_vals[Q_vals_ep,state,:] = qvs
                    else:
                        obs = self.base_env.observation(state)
                        qvs = self.agent.get_Q_vals(obs)
                        Q_vals[Q_vals_ep,state,:] = qvs
                Q_vals_ep += 1

            # Estimate statistics of the replay buffer contents.
            if (episode > 1) and (episode % replay_buffer_counts_period == 0):
                replay_buffer_counts_episodes.append(episode)
                replay_buffer_counts.append(self.agent.get_replay_buffer_counts())

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
        data['Q_vals_episodes'] = Q_vals_episodes
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)
        data['policy'] = np.argmax(Q_vals[-1], axis=1)
        data['replay_buffer_counts_episodes'] = replay_buffer_counts_episodes
        data['replay_buffer_counts'] = replay_buffer_counts
        data['rollouts_episodes'] = rollouts_episodes
        data['rollouts_rewards'] = rollouts_rewards

        return data

    def _create_dataset(self):
        print('Creating static dataset of transitions...')

        static_dataset = []
        dataset_size = self.synthetic_static_dataset_size * self.base_env.time_limit
        mesh = np.array(np.meshgrid(np.arange(self.base_env.num_states),
                                    np.arange(self.base_env.num_actions)))
        sa_combinations = mesh.T.reshape(-1, 2)

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

            observation = self.base_env.observation(state)

            # Sample next state, observation and reward.
            self.base_env.set_state(state)
            next_observation, reward, done, info = self.base_env.step(action)

            transition = (observation, np.array(action, dtype=np.int32),
                            np.array(reward, dtype=np.float32),
                            np.array(1.0, dtype=np.float32),
                            next_observation)
            extras = (np.int32(state),)

            static_dataset.append((transition, extras))

            self.base_env.reset()

        print(f'Static dataset created containing {len(static_dataset)} transitions.')

        return static_dataset

    def _execute_rollout(self):

        timestep = self.env.reset()

        episode_cumulative_reward = 0
        while not timestep.last():

            action = self.agent.select_action(timestep.observation)
            timestep = self.env.step(action)
            episode_cumulative_reward += timestep.reward

        return episode_cumulative_reward
