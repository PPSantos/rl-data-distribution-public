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

from algos.oracle_fqi import oracle_fqi_acme
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


class OracleFQI(object):

    def __init__(self, env, env_grid_spec, oracle_fqi_args):

        self.base_env = env
        self.env_grid_spec = env_grid_spec
        self.env = wrap_env(env)
        env_spec = acme.make_environment_spec(self.env)

        network = _make_network(env_spec,
                        hidden_layers=oracle_fqi_args['hidden_layers'])

        self.agent = oracle_fqi_acme.OracleFQI(environment_spec=env_spec,
                                    network=network,
                                    batch_size=oracle_fqi_args['batch_size'],
                                    prefetch_size=oracle_fqi_args['prefetch_size'],
                                    num_sampling_steps=oracle_fqi_args['num_sampling_steps'],
                                    num_gradient_steps=oracle_fqi_args['num_gradient_steps'],
                                    max_replay_size=oracle_fqi_args['max_replay_size'],
                                    n_step=oracle_fqi_args['n_step'],
                                    epsilon_init=oracle_fqi_args['epsilon_init'],
                                    epsilon_final=oracle_fqi_args['epsilon_final'],
                                    epsilon_schedule_timesteps=oracle_fqi_args['epsilon_schedule_timesteps'],
                                    learning_rate=oracle_fqi_args['learning_rate'],
                                    discount=oracle_fqi_args['discount'],
                                    reweighting_type=oracle_fqi_args['reweighting_type'],
                                    num_states=self.base_env.num_states,
                                    num_actions=self.base_env.num_actions)

        self.oracle_q_vals = oracle_fqi_args['oracle_q_vals']

    def train(self, num_episodes):

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

                oracle_q_val = np.float32(self.oracle_q_vals[env_state,action])
                env_state = np.int32(env_state)
                self.agent.observe_with_extras(action, next_timestep=timestep, extras=(env_state, oracle_q_val))

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
                replay_buffer_counts.append(self.agent.estimate_replay_buffer_counts())

        data = {}
        data['episode_rewards'] = episode_rewards
        data['states_counts'] = states_counts
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)
        data['policy'] = np.argmax(Q_vals[-1], axis=1)
        data['replay_buffer_counts'] = replay_buffer_counts

        return data
