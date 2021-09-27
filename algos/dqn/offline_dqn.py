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


class OfflineDQN(object):

    def __init__(self, env, env_grid_spec, log_path, offline_dqn_args):

        np.random.seed()

        self._base_env = env
        self._env_grid_spec = env_grid_spec
        self._env = wrap_env(env)
        env_spec = acme.make_environment_spec(self._env)

        network = _make_network(env_spec,
                        hidden_layers=offline_dqn_args['hidden_layers'])

        self._agent = dqn_acme.DQN(environment_spec=env_spec,
                                  network=network,
                                  batch_size=offline_dqn_args['batch_size'],
                                  target_update_period=offline_dqn_args['target_update_period'],
                                  max_replay_size=offline_dqn_args['dataset_size'],
                                  learning_rate=offline_dqn_args['learning_rate'],
                                  discount=offline_dqn_args['discount'],
                                  num_states=self._base_env.num_states,
                                  num_actions=self._base_env.num_actions,
                                  logger=loggers.CSVLogger(
                                        directory_or_file=log_path, label='learner'))

        # Internalise arguments.
        self._num_learning_steps = offline_dqn_args['num_learning_steps']

        # Synthesize dasaset using sampling distribution and pre-fill the replay buffer.
        self._sampling_dist_size = self._base_env.num_states * self._base_env.num_actions
        if offline_dqn_args['dataset_custom_sampling_dist'] is None:
            self._sampling_dist = np.random.dirichlet(
                            [offline_dqn_args['dataset_sampling_dist_alpha']]*self._sampling_dist_size)
        else:
            # Load custom sampling distribution.
            custom_sampling_dist_path = offline_dqn_args['dataset_custom_sampling_dist']
            print(f'Loading custom sampling dist from {custom_sampling_dist_path}')    
            with open(custom_sampling_dist_path, 'r') as f:
                sampling_dist_data = json.load(f)
                sampling_dist_data = json.loads(sampling_dist_data)
            f.close()
            self._sampling_dist = sampling_dist_data['sampling_dist']

        print('self._sampling_dist:', self._sampling_dist)
        print('self._sampling_dist_size:', self._sampling_dist_size)

        self._create_dataset(offline_dqn_args['dataset_size'],
                offline_dqn_args['dataset_force_full_coverage'])

    def train(self, q_vals_period, rollouts_period, num_rollouts,
                rollouts_envs, replay_buffer_counts_period):

        rollouts_envs = [wrap_env(e) for e in rollouts_envs]

        Q_vals = np.zeros((self._num_learning_steps//q_vals_period,
                self._base_env.num_states, self._base_env.num_actions))
        Q_vals_steps = []
        Q_vals_idx = 0

        rollouts_steps = []
        rollouts_rewards = []

        replay_buffer_counts_steps = []
        replay_buffer_counts = []

        for step in tqdm(range(self._num_learning_steps)):

            # Update learner.
            self._agent.learner_step()

            # Store current Q-values.
            if step % q_vals_period == 0:
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

                Q_vals_steps.append(step)
                Q_vals[Q_vals_idx,:,:] = estimated_Q_vals
                Q_vals_idx += 1

            # Execute evaluation rollouts.
            if step % rollouts_period == 0:
                print('Executing evaluation rollouts.')

                r_rewards = []
                for r_env in rollouts_envs:
                    r_rewards.append([self._execute_rollout(r_env) for _ in range(num_rollouts)])

                rollouts_steps.append(step)
                rollouts_rewards.append(r_rewards)

            # Get replay buffer statistics.
            if (step > 1) and (step % replay_buffer_counts_period == 0):
                print('Getting replay buffer statistics.')
                replay_buffer_counts_steps.append(step)
                replay_buffer_counts.append(self._agent.get_replay_buffer_counts())

        data = {}
        data['Q_vals'] = Q_vals
        data['Q_vals_steps'] = Q_vals_steps
        data['rollouts_rewards'] = rollouts_rewards
        data['rollouts_steps'] = rollouts_steps
        data['replay_buffer_counts'] = replay_buffer_counts
        data['replay_buffer_counts_steps'] = replay_buffer_counts_steps

        return data

    def _create_dataset(self, dataset_size, force_full_coverage):
        print('Pre-filling replay buffer.')

        mesh = np.array(np.meshgrid(np.arange(self._base_env.num_states),
                                    np.arange(self._base_env.num_actions)))
        sa_combinations = mesh.T.reshape(-1, 2)
        sa_counts = np.zeros((self._base_env.num_states, self._base_env.num_actions))

        for _ in range(dataset_size):

            # Randomly sample (state, action) pair according to sampling dist.
            if self._env_grid_spec:
                tile_type = TileType.WALL
                while tile_type == TileType.WALL:
                    sampled_idx = np.random.choice(np.arange(self._sampling_dist_size), p=self._sampling_dist)
                    state, action = sa_combinations[sampled_idx]
                    xy = self._env_grid_spec.idx_to_xy(state)
                    tile_type = self._env_grid_spec.get_value(xy)
            else:
                sampled_idx = np.random.choice(np.arange(self._sampling_dist_size), p=self._sampling_dist)
                state, action = sa_combinations[sampled_idx]

            sa_counts[state,action] += 1
            observation = self._base_env.observation(state)

            # Sample next state, observation and reward.
            self._base_env.set_state(state)
            next_observation, reward, done, info = self._base_env.step(action)

            # Add to replay buffer.
            transition = (observation, np.array(action, dtype=np.int32),
                            np.array(reward, dtype=np.float32),
                            np.array(1.0, dtype=np.float32),
                            next_observation)
            extras = (np.int32(state), np.int32(self._base_env.get_state()))
            self._agent.add_to_replay_buffer(transition, extras)

            self._base_env.reset()

        if force_full_coverage:
            # Correct dataset such that we have coverage over all (state, action) pairs.
            zero_positions = np.where(sa_counts == 0)
            print('Number of missing (s,a) pairs:', np.sum((sa_counts == 0)))
            for (state, action) in zip(*zero_positions):
                
                # Skip walls.
                if self._env_grid_spec:
                    xy = self._env_grid_spec.idx_to_xy(state)
                    tile_type = self._env_grid_spec.get_value(xy)
                    if tile_type == TileType.WALL:
                        continue

                observation = self._base_env.observation(state)

                # Sample next state, observation and reward.
                self._base_env.set_state(state)
                next_observation, reward, done, info = self._base_env.step(action)

                # Add to replay buffer.
                transition = (observation, np.array(action, dtype=np.int32),
                                np.array(reward, dtype=np.float32),
                                np.array(1.0, dtype=np.float32),
                                next_observation)
                extras = (np.int32(state), np.int32(self._base_env.get_state()))
                self._agent.add_to_replay_buffer(transition, extras)

                self._base_env.reset()

    def _execute_rollout(self, r_env):
        timestep = r_env.reset()
        rollout_cumulative_reward = 0
        while not timestep.last():
            action = self._agent.deterministic_action(timestep.observation)
            timestep = r_env.step(action)
            rollout_cumulative_reward += timestep.reward

        return rollout_cumulative_reward
