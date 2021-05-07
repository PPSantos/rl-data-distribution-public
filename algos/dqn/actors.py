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

"""Actor implementation with "extras"."""

from typing import Optional, Tuple

from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class FeedForwardActor(core.Actor):
    """A feed-forward actor with "extras".
    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        replay_buffer = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the actor.
        Args:
        policy_network: the policy to run.
        replay_buffer: the replay buffer object to which allows to add experiences to a
            dataset/replay buffer.
        variable_client: object which allows to copy weights from the learner copy
            of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._replay_buffer = replay_buffer
        self._variable_client = variable_client
        self._policy_network = policy_network

        self._prev_timestep = None

    @tf.function
    def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

        return action

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._replay_buffer:
            self._prev_timestep = timestep

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):

        if self._replay_buffer:

            prev_observation = tf.constant(self._prev_timestep.observation, dtype=tf.float32)
            action = tf.constant(action, dtype=tf.int32)
            reward = tf.constant(next_timestep.reward, dtype=tf.float32)
            discount = tf.constant(next_timestep.discount, dtype=tf.float32)
            observation = tf.constant(next_timestep.observation, dtype=tf.float32)

            values = (prev_observation, action, reward, discount, observation)
            transition = tf.nest.map_structure(lambda t: tf.stack([t]),
                                         values)

            self._replay_buffer.add_batch(transition)

    # def observe_with_extras(self, action: types.NestedArray, next_timestep: dm_env.TimeStep, extras=None):
    #     # Allows to add extras to the replay buffer.
    #     if self._adder:
    #         if extras:
    #             self._adder.add(action, next_timestep, extras=extras)
    #         else:
    #             self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)
