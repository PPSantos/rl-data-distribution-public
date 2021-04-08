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
                  torso_layers : Sequence[int] = [5],
                  head_layers  : Sequence[int] = [5]):
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

        print('dqn env_spec:', env_spec)

        network = _make_network(env_spec)

        self.agent = dqn_acme.DQN(environment_spec=env_spec,
                                    network=network, **dqn_args)

    def train(self, num_episodes):

        for episode in tqdm(range(num_episodes)):

            timestep = self.env.reset()
            self.agent.observe_first(timestep)

            while not timestep.last():

                action = self.agent.select_action(timestep.observation)
                timestep = self.env.step(action)

                self.agent.observe(action, next_timestep=timestep)

                self.agent.update()

        # Calculate policy.
        Q = np.zeros((self.base_env.num_states, self.base_env.num_actions))
        policy = {}
        max_Q_vals = {}
        for state in range(self.base_env.num_states):

            obs = self.base_env.observation(state)
            q_vals = self.agent.get_Q_vals(obs)

            Q[state,:] = q_vals
            policy[state] = np.argmax(q_vals)
            max_Q_vals[state] = np.max(q_vals)

        return Q, max_Q_vals, policy
