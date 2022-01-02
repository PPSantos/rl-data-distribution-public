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

        self._base_env = env
        self._env_grid_spec = env_grid_spec
        self._env = wrap_env(env)
        env_spec = acme.make_environment_spec(self._env)

        network = _make_network(env_spec,
                        hidden_layers=dqn_args['hidden_layers'])

        self._agent = dqn_acme.DQN(environment_spec=env_spec,
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
                                    num_states=self._base_env.num_states,
                                    num_actions=self._base_env.num_actions,
                                    logger=loggers.CSVLogger(directory_or_file=log_path, label='learner'))
        
        # Internalise arguments.
        self._num_learning_episodes = dqn_args['num_learning_episodes']

    def train(self, q_vals_period, replay_buffer_counts_period,
                num_rollouts, rollouts_period, rollouts_envs):

        rollouts_envs = [wrap_env(e) for e in rollouts_envs]

        episode_rewards = []

        Q_vals = np.zeros((self._num_learning_episodes//q_vals_period,
                self._base_env.num_states, self._base_env.num_actions))
        Q_vals_episodes = []
        Q_vals_idx = 0

        rollouts_episodes = []
        rollouts_rewards = []

        replay_buffer_counts_episodes = []
        replay_buffer_counts = []

        for episode in tqdm(range(self._num_learning_episodes)):

            timestep = self._env.reset()
            env_state = self._base_env.get_state()

            self._agent.observe_first(timestep)

            episode_cumulative_reward = 0
            while not timestep.last():

                action = self._agent.select_action(timestep.observation)
                timestep = self._env.step(action)

                self._agent.observe_with_extras(action,
                    next_timestep=timestep, extras=(np.int32(env_state),
                                np.int32(self._base_env.get_state())))

                self._agent.update()

                env_state = self._base_env.get_state()

                # Log data.
                episode_cumulative_reward += timestep.reward

            episode_rewards.append(episode_cumulative_reward)

            # Store current Q-values.
            if episode % q_vals_period == 0:
                print('Storing current Q-values estimates.')
                estimated_Q_vals = np.zeros((self._base_env.num_states, self._base_env.num_actions))
                for state in range(self._base_env.num_states):
                    if self._env_grid_spec:
                        xy = self._env_grid_spec.idx_to_xy(state)
                        tile_type = self._env_grid_spec.get_value(xy)
                        if tile_type == TileType.WALL:
                            estimated_Q_vals[state,:] = 0
                        else:
                            obs = self._base_env.observation(state)
                            qvs = self._agent.get_Q_vals(obs)
                            estimated_Q_vals[state,:] = qvs
                    else:
                        obs = self._base_env.observation(state)
                        qvs = self._agent.get_Q_vals(obs)
                        estimated_Q_vals[state,:] = qvs

                Q_vals_episodes.append(episode)
                Q_vals[Q_vals_idx,:,:] = estimated_Q_vals
                Q_vals_idx += 1

            # Get replay buffer statistics.
            if (episode > 1) and (episode % replay_buffer_counts_period == 0):
                print('Getting replay buffer statistics.')
                replay_buffer_counts_episodes.append(episode)
                replay_buffer_counts.append(self._agent.get_replay_buffer_counts())

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
        data['Q_vals_episodes'] = Q_vals_episodes
        data['Q_vals'] = Q_vals
        data['replay_buffer_counts_episodes'] = replay_buffer_counts_episodes
        data['replay_buffer_counts'] = replay_buffer_counts
        data['rollouts_episodes'] = rollouts_episodes
        data['rollouts_rewards'] = rollouts_rewards

        return data

    def _execute_rollout(self, r_env):
        timestep = r_env.reset()
        rollout_cumulative_reward = 0
        while not timestep.last():
            action = self._agent.deterministic_action(timestep.observation)
            timestep = r_env.step(action)
            rollout_cumulative_reward += timestep.reward

        return rollout_cumulative_reward
