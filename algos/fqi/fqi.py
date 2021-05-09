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

from algos.fqi import fqi_acme
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


class FQI(object):

    def __init__(self, env, env_grid_spec, fqi_args):

        self.base_env = env
        self.env_grid_spec = env_grid_spec
        self.env = wrap_env(env)
        env_spec = acme.make_environment_spec(self.env)

        network = _make_network(env_spec,
                        hidden_layers=fqi_args['hidden_layers'])

        self.agent = fqi_acme.FQI(environment_spec=env_spec,
                                    network=network,
                                    batch_size=fqi_args['batch_size'],
                                    prefetch_size=fqi_args['prefetch_size'],
                                    num_sampling_steps=fqi_args['num_sampling_steps'],
                                    num_gradient_steps=fqi_args['num_gradient_steps'],
                                    max_replay_size=fqi_args['max_replay_size'],
                                    n_step=fqi_args['n_step'],
                                    epsilon_init=fqi_args['epsilon_init'],
                                    epsilon_final=fqi_args['epsilon_final'],
                                    epsilon_schedule_timesteps=fqi_args['epsilon_schedule_timesteps'],
                                    learning_rate=fqi_args['learning_rate'],
                                    discount=fqi_args['discount'],
                                    reweighting_type=fqi_args['reweighting_type'],
                                    uniform_replay_buffer=fqi_args['uniform_replay_buffer'],
                                    num_states=self.base_env.num_states,
                                    num_actions=self.base_env.num_actions)

        self.uniform_replay_buffer = fqi_args['uniform_replay_buffer']
        self.uniform_static_dataset_size = fqi_args['uniform_static_dataset_size']

    def train(self, num_episodes):

        if self.uniform_replay_buffer:
            static_dataset = self.create_static_uniform_dataset()

        states_counts = np.zeros((self.env.num_states))
        episode_rewards = []
        Q_vals = np.zeros((num_episodes, self.base_env.num_states, self.base_env.num_actions))
        replay_buffer_counts = []

        for episode in tqdm(range(num_episodes)):

            timestep = self.env.reset()
            self.agent.observe_first(timestep)
            env_state = self.base_env.get_state()

            episode_cumulative_reward = 0
            while not timestep.last():

                action = self.agent.select_action(timestep.observation)
                timestep = self.env.step(action)

                env_state = np.int32(env_state)
                self.agent.observe_with_extras(action, next_timestep=timestep, extras=(env_state,))

                if self.uniform_replay_buffer:
                    # Insert a randomly selected transition from the static dataset.
                    idx = np.random.randint(self.uniform_static_dataset_size)
                    transition, extras = static_dataset[idx]
                    self.agent.add_to_replay_buffer(transition, extras)

                self.agent.update()

                env_state = self.base_env.get_state()

                # Log data.
                episode_cumulative_reward += timestep.reward
                states_counts[self.base_env.get_state()] += 1

            episode_rewards.append(episode_cumulative_reward)

            # Store current Q-values (filters wall states).
            for state in range(self.base_env.num_states):
                xy = self.env_grid_spec.idx_to_xy(state)
                tile_type = self.env_grid_spec.get_value(xy)
                if tile_type == TileType.WALL:
                    Q_vals[episode,state,:] = 0
                else:
                    obs = self.base_env.observation(state)
                    qvs = self.agent.get_Q_vals(obs)
                    Q_vals[episode,state,:] = qvs

            if (episode > 1) and (episode % 500 == 0):
                # Estimate statistics of the replay buffer contents.
                replay_buffer_counts.append(self.agent.get_replay_buffer_counts())

        data = {}
        data['episode_rewards'] = episode_rewards
        data['states_counts'] = states_counts
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)
        data['policy'] = np.argmax(Q_vals[-1], axis=1)
        data['replay_buffer_counts'] = replay_buffer_counts

        return data

    def create_static_uniform_dataset(self):
        print('Creating static dataset with uniformly sampled transitions...')

        static_dataset = []

        for _ in range(self.uniform_static_dataset_size):
            
            # Randomly uniform sample state.
            tile_type = TileType.WALL
            while tile_type == TileType.WALL:
                state = np.random.randint(self.base_env.num_states)
                xy = self.env_grid_spec.idx_to_xy(state)
                tile_type = self.env_grid_spec.get_value(xy)

            observation = self.base_env.observation(state)

            # Randomly uniform sample action.
            action = np.random.randint(self.base_env.num_actions)

            # Sample next state, observation and reward.
            self.base_env.set_state(state)
            next_observation, reward, done, info = self.base_env.step(action)

            transition = (observation, action, reward, 1.0, next_observation)
            extras = (state,)

            static_dataset.append((transition, extras))

            self.base_env.reset()

        print(f'Static uniform dataset created containing {len(static_dataset)} transitions.')

        return static_dataset
