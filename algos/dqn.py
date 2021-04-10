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

from algos import dqn_acme
#from ilurl.utils.default_logger import make_default_logger


def _make_network(env_spec : dm_env,
                  torso_layers : Sequence[int] = [10],
                  head_layers  : Sequence[int] = [10]):
    network = snt.Sequential([
        # Torso MLP.
        snt.nets.MLP(torso_layers, activate_final=True),
        # Dueling MLP head.
        networks.DuellingMLP(num_actions=env_spec.actions.num_values,
                            hidden_sizes=head_layers)  
    ])
    return network

def wrap_env(env):
    return wrappers.wrap_all(env, [
        wrappers.GymWrapper,
        wrappers.SinglePrecisionWrapper,
    ])


class DQN(object):

    def __init__(self, env, dqn_args):

        self.base_env = env
        self.env = wrap_env(env)
        env_spec = acme.make_environment_spec(self.env)

        network = _make_network(env_spec)

        self.agent = dqn_acme.DQN(environment_spec=env_spec,
                                    network=network, **dqn_args)

    def train(self, num_episodes):

        states_counts = np.zeros((self.env.num_states))
        episode_rewards = []
        Q_vals = np.zeros((num_episodes, self.base_env.num_states, self.base_env.num_actions))

        for episode in tqdm(range(num_episodes)):

            timestep = self.env.reset()
            self.agent.observe_first(timestep)

            episode_cumulative_reward = 0
            while not timestep.last():

                action = self.agent.select_action(timestep.observation)
                timestep = self.env.step(action)

                self.agent.observe(action, next_timestep=timestep)

                self.agent.update()

                # Log data.
                episode_cumulative_reward += timestep.reward
                states_counts[self.base_env.get_state()] += 1

            episode_rewards.append(episode_cumulative_reward)

            # Store current Q-values.
            for state in range(self.base_env.num_states):
                obs = self.base_env.observation(state)
                qvs = self.agent.get_Q_vals(obs)
                Q_vals[episode,state,:] = qvs

        # Calculate policy.
        policy = {}
        max_Q_vals = {}
        for state in range(self.base_env.num_states):
            obs = self.base_env.observation(state)
            qvs = self.agent.get_Q_vals(obs)
            policy[state] = np.argmax(qvs)
            max_Q_vals[state] = np.max(qvs)

        data = {}
        data['episode_rewards'] = episode_rewards
        data['epsilon_values'] = None
        data['states_counts'] = states_counts
        data['Q_vals'] = Q_vals
        data['max_Q_vals'] = max_Q_vals
        data['policy'] = policy

        return data
