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

"""OracleFQILearner learner implementation (actions re-weighting)."""

import time
from typing import Dict, List

import acme
from acme import types
from acme.adders import reverb as adders
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class OracleFQILearnerReweightActions(acme.Learner, tf2_savers.TFSaveable):
    """OracleFQILearner unprioritized learner (actions re-weighting).

    This is the learning component of a OracleFQILearner agent. It takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        network: snt.Module,
        target_network: snt.Module,
        discount: float,
        learning_rate: float,
        dataset: tf.data.Dataset,
        huber_loss_parameter: float = 1.,
        replay_client: reverb.TFClient = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        max_gradient_norm: float = None,
        num_states: int = None,
        num_actions : int = None,
    ):
        """Initializes the learner.

        Args:
        network: the online Q network (the one being optimized)
        target_network: the target Q critic (which lags behind the online net).
        discount: discount to use for TD updates.
        learning_rate: learning rate for the q-network update.
        dataset: dataset to learn from, whether fixed or from a replay buffer (see
            `acme.datasets.reverb.make_dataset` documentation).
        huber_loss_parameter: Quadratic-linear boundary for Huber loss.
        replay_client: client to replay to allow for updating priorities.
        counter: Counter object for (potentially distributed) counting.
        logger: Logger object for writing logs to.
        checkpoint: boolean indicating whether to checkpoint the learner.
        max_gradient_norm: used for gradient clipping.
        """
        # Internalise agent components (replay buffer, networks, optimizer).
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self._network = network
        self._target_network = target_network
        self._optimizer = snt.optimizers.Adam(learning_rate)
        self._replay_client = replay_client

        # Internalise the hyperparameters.
        self._discount = discount
        self._huber_loss_parameter = huber_loss_parameter
        if max_gradient_norm is None:
            max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
        self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

        # Learner state.
        self._variables: List[List[tf.Tensor]] = [network.trainable_variables]
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Internalise logging/counting objects.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        # Create a snapshotter object.
        if checkpoint:
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={'network': network}, time_delta_minutes=60.)
        else:
            self._snapshotter = None

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

        self.num_states = num_states
        self.num_actions = num_actions
        self._replay_buffer_counts = tf.Variable(np.zeros((num_states,num_actions)))

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:

        inputs = next(self._iterator)
        transitions: types.Transition = inputs.data
        targets = inputs.data.extras['oracle_q_vals']
        states = inputs.data.extras['env_state']

        # Calculate importance weights.
        summed = tf.reduce_sum(self._replay_buffer_counts, axis=1, keepdims=True)
        p_a_s = tf.divide(self._replay_buffer_counts, summed)

        idxs = tf.stack([states, transitions.action], axis=1)
        p_a_s = tf.gather_nd(p_a_s, idxs)
        weights = tf.divide((1/self.num_actions), p_a_s)

        with tf.GradientTape() as tape:

            q_tm1 = self._network(transitions.observation) # [B,A]
            qa_tm1 = trfl.indexing_ops.batched_index(q_tm1, transitions.action) # [B]

            error = targets - qa_tm1 # [B]
            loss = losses.huber(error, self._huber_loss_parameter)
            loss *= tf.cast(weights, loss.dtype)
            loss = tf.reduce_mean(loss, axis=[0])  # []

        # Do a step of SGD.
        gradients = tape.gradient(loss, self._network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._optimizer.apply(gradients, self._network.trainable_variables)

        self._num_steps.assign_add(1)

        # Report loss & statistics for logging.
        fetches = {
            'loss': loss,
        }

        return fetches

    def update_target_network(self):
        # Update the target network.
        for src, dest in zip(self._network.variables,
                            self._target_network.variables):
            dest.assign(src)

    def update_replay_buffer_counts(self, state, action):
        self._replay_buffer_counts[state,action].assign(self._replay_buffer_counts[state,action] + 1)

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        result.update(counts)

        # Snapshot and attempt to write logs.
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(result)

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        return tf2_utils.to_numpy(self._variables)

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'network': self._network,
            'target_network': self._target_network,
            'optimizer': self._optimizer,
            'num_steps': self._num_steps
        }