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
from acme.utils import loggers

from algos.dqn2be import dqn2be_acme
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


class DQN2BE(object):

    def __init__(self, env, env_grid_spec, log_path, dqn2be_args):

        np.random.seed()

        self.base_env = env
        self.env_grid_spec = env_grid_spec
        self.env = wrap_env(env)
        env_spec = acme.make_environment_spec(self.env)

        network = _make_network(env_spec,
                        hidden_layers=dqn2be_args['hidden_layers'])
        e_network = _make_network(env_spec,
                        hidden_layers=dqn2be_args['e_net_hidden_layers'])

        self.agent = dqn2be_acme.DQN2BE(environment_spec=env_spec,
                                    network=network,
                                    e_network=e_network,
                                    batch_size=dqn2be_args['batch_size'],
                                    target_update_period=dqn2be_args['target_update_period'],
                                    target_e_net_update_period=dqn2be_args['target_e_net_update_period'],
                                    samples_per_insert=dqn2be_args['samples_per_insert'],
                                    min_replay_size=dqn2be_args['min_replay_size'],
                                    max_replay_size=dqn2be_args['max_replay_size'],
                                    prioritized_replay=dqn2be_args['prioritized_replay'],
                                    importance_sampling_exponent=dqn2be_args['importance_sampling_exponent'],
                                    priority_exponent=dqn2be_args['priority_exponent'],
                                    epsilon_init=dqn2be_args['epsilon_init'],
                                    epsilon_final=dqn2be_args['epsilon_final'],
                                    epsilon_schedule_timesteps=dqn2be_args['epsilon_schedule_timesteps'],
                                    learning_rate=dqn2be_args['learning_rate'],
                                    e_net_learning_rate=dqn2be_args['e_net_learning_rate'],
                                    discount=dqn2be_args['discount'],
                                    # delta_init=dqn2be_args['delta_init'],
                                    # delta_final=dqn2be_args['delta_final'],
                                    # delta_schedule_timesteps=dqn2be_args['delta_schedule_timesteps'],
                                    num_states=self.base_env.num_states,
                                    num_actions=self.base_env.num_actions,
                                    logger=loggers.CSVLogger(directory_or_file=log_path, label='learner'))

    def train(self, num_episodes, q_vals_period, replay_buffer_counts_period,
                num_rollouts, rollouts_period, rollouts_envs):

        rollouts_envs = [wrap_env(e) for e in rollouts_envs]

        states_counts = np.zeros((self.env.num_states))
        episode_rewards = []

        Q_vals = np.zeros((num_episodes//q_vals_period,
                self.base_env.num_states, self.base_env.num_actions))
        Q_vals_episodes = []
        Q_vals_ep = 0
        E_vals = np.zeros((num_episodes//q_vals_period,
                self.base_env.num_states, self.base_env.num_actions))

        replay_buffer_counts_episodes = []
        replay_buffer_counts = []

        rollouts_episodes = []
        rollouts_rewards = []

        for episode in tqdm(range(num_episodes)):

            timestep = self.env.reset()
            env_state = self.base_env.get_state()

            self.agent.observe_first(timestep)

            episode_cumulative_reward = 0
            while not timestep.last():

                action = self.agent.select_action(timestep.observation)
                timestep = self.env.step(action)

                self.agent.observe_with_extras(action,
                    next_timestep=timestep, extras=(np.int32(env_state),))

                self.agent.update()

                env_state = self.base_env.get_state()

                # Log data.
                episode_cumulative_reward += timestep.reward
                states_counts[self.base_env.get_state()] += 1

            episode_rewards.append(episode_cumulative_reward)

            # Store current Q-values and E-values.
            if episode % q_vals_period == 0:
                Q_vals_episodes.append(episode)
                for state in range(self.base_env.num_states):
                    if self.env_grid_spec:
                        xy = self.env_grid_spec.idx_to_xy(state)
                        tile_type = self.env_grid_spec.get_value(xy)
                        if tile_type == TileType.WALL:
                            Q_vals[Q_vals_ep,state,:] = 0
                            E_vals[Q_vals_ep,state,:] = 0
                        else:
                            obs = self.base_env.observation(state)
                            qvs = self.agent.get_Q_vals(obs)
                            Q_vals[Q_vals_ep,state,:] = qvs
                            evs = self.agent.get_E_vals(obs)
                            E_vals[Q_vals_ep,state,:] = evs
                    else:
                        obs = self.base_env.observation(state)
                        qvs = self.agent.get_Q_vals(obs)
                        Q_vals[Q_vals_ep,state,:] = qvs
                        evs = self.agent.get_E_vals(obs)
                        E_vals[Q_vals_ep,:] = evs
                Q_vals_ep += 1

            # Get replay buffer statistics.
            if (episode > 1) and (episode % replay_buffer_counts_period == 0):
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
        data['E_vals'] = E_vals
        data['max_Q_vals'] = np.max(Q_vals[-1], axis=1)
        data['policy'] = np.argmax(Q_vals[-1], axis=1)
        data['replay_buffer_counts_episodes'] = replay_buffer_counts_episodes
        data['replay_buffer_counts'] = replay_buffer_counts
        data['rollouts_episodes'] = rollouts_episodes
        data['rollouts_rewards'] = rollouts_rewards

        return data

    def _execute_rollout(self, r_env):

        timestep = r_env.reset()

        rollout_cumulative_reward = 0
        while not timestep.last():
            action = self.agent.deterministic_action(timestep.observation)
            timestep = r_env.step(action)
            rollout_cumulative_reward += timestep.reward

        return rollout_cumulative_reward
