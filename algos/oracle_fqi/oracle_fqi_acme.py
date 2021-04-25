# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracle FQI agent implementation."""

import copy
from typing import Optional

import numpy as np

from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from algos.oracle_fqi.fqi_agent_acme import FQIAgent
from acme.agents.tf.dqn import learning
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import reverb
import sonnet as snt
import tensorflow as tf
import trfl

from algos.oracle_fqi import actors

from algos.utils import tf2_savers
from algos.utils.tf2_layers import EpsilonGreedyExploration
from algos.oracle_fqi.oracle_fqi_acme_learning import OracleFQILearner


class OracleFQI(FQIAgent):
    """
        OracleFQI agent.
    """

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            network: snt.Module,
            batch_size: int = 256,
            prefetch_size: int = 4,
            num_sampling_steps: int = 1000,
            num_gradient_steps: int = 10,
            max_replay_size: int = 10000,
            n_step: int = 1,
            epsilon_init: float = 1.0,
            epsilon_final: float = 0.01,
            epsilon_schedule_timesteps: int = 20000,
            learning_rate: float = 1e-3,
            discount: float = 0.99,
            max_gradient_norm: Optional[float] = None,
            logger: loggers.Logger = None,
        ):
        """Initialize the agent.
        Args:
        environment_spec: description of the actions, observations, etc.
        network: the online Q network (the one being optimized).
        batch_size: batch size for updates.
        prefetch_size: size to prefetch from replay.
        num_sampling_steps: number of sampling steps to perform.
        num_gradient_steps: number of gradient descent steps to perform.
        max_replay_size: maximum size of replay buffer.
        n_step: number of steps to squash into a single transition.
        epsilon_init: Initial epsilon value (probability of taking a random action)
        epsilon_final: Final epsilon value (probability of taking a random action)
        epsilon_schedule_timesteps: timesteps to decay epsilon from 'epsilon_init'
            to 'epsilon_final'. 
        learning_rate: learning rate for the q-network update.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.
        """

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec,
                                                extras_spec={'oracle_q_vals': np.float32(1.0)}))
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        replay_client = reverb.TFClient(address)
        dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=batch_size,
            prefetch_size=prefetch_size)

        policy_network = snt.Sequential([
            network,
            EpsilonGreedyExploration(epsilon_init=epsilon_init,
                                     epsilon_final=epsilon_final,
                                     epsilon_schedule_timesteps=epsilon_schedule_timesteps)
        ])

        # Create a target network.
        target_network = copy.deepcopy(network)

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(network, [environment_spec.observations])
        tf2_utils.create_variables(target_network, [environment_spec.observations])

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(policy_network, adder)

        # The learner updates the parameters (and initializes them).
        learner = OracleFQILearner(
            network=network,
            target_network=target_network,
            discount=discount,
            learning_rate=learning_rate,
            # target_update_period=target_update_period, # target network is updated by superclass.
            dataset=dataset,
            replay_client=replay_client,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            checkpoint=False)

        self._saver = tf2_savers.Saver(learner.state)

        # Deterministic (max-Q) actor.
        max_Q_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=0.0).sample(),
        ])
        self._deterministic_actor = actors.FeedForwardActor(max_Q_network)

        self._Q_vals_actor = actors.FeedForwardActor(network)

        super().__init__(
            actor=actor,
            learner=learner,
            num_sampling_steps=num_sampling_steps,
            num_gradient_steps=num_gradient_steps)

    def update(self):
        super().update()

    def get_Q_vals(self, obs: types.NestedArray) -> types.NestedArray:
        return self._Q_vals_actor.select_action(obs)

    def deterministic_action(self, obs: types.NestedArray) -> types.NestedArray:
        return self._deterministic_actor.select_action(obs)

    def save(self, p):
        self._saver.save(p)

    def load(self, p):
        self._saver.load(p)
