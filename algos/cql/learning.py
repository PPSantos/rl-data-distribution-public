"""
    CQL learner implementation.
"""
import time
from typing import Dict, List, Optional

import trfl
import numpy as np
import sonnet as snt
import tensorflow as tf

import acme
from acme import types
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers

from utils import tf2_savers

class CQLLearner(acme.Learner, tf2_savers.TFSaveable):
    """CQL learner.

    This is the learning component of a CQL agent. It takes a dataset as input
    and implements update functionality to learn from this dataset. It colapses
    to a regular DQN when alpha = 0.
    """
    def __init__(
        self,
        network: snt.Module,
        target_network: snt.Module,
        discount: float,
        learning_rate: float,
        alpha: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        huber_loss_parameter: float = 1.,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_interval: int = 5_000,
        save_directory: str = '~/acme',
        max_gradient_norm: Optional[float] = None,
    ):
        """Initializes the learner.

        Args:
            network: the online Q network (the one being optimized)
            target_network: the target Q critic (which lags behind the online net).
            discount: discount to use for TD updates.
            learning_rate: learning rate for the q-network update.
            alpha: CQL reguralizer penalty.
            target_update_period: number of learner steps to perform before updating
            the target networks.
            dataset: dataset to learn from, whether fixed or from a replay buffer (see
            `acme.datasets.reverb.make_dataset` documentation).
            huber_loss_parameter: Quadratic-linear boundary for Huber loss.
            counter: Counter object for (potentially distributed) counting.
            logger: Logger object for writing logs to.
            checkpoint: boolean indicating whether to checkpoint the learner.
            checkpoint_interval: interval at which to checkpoint the learner.
            save_directory: string indicating where the learner should save
            checkpoints and snapshots.
            max_gradient_norm: used for gradient clipping.
        """

        # Internalise agent components (replay buffer, networks, optimizer).
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self._network = network
        self._target_network = target_network
        self._optimizer = snt.optimizers.Adam(learning_rate)

        # Make sure to initialize the optimizer so that its variables (e.g. the Adam
        # moments) are included in the state returned by the learner (which can then
        # be checkpointed and restored).
        self._optimizer._initialize(network.trainable_variables)  # pylint: disable= protected-access

        # Internalise the hyperparameters.
        self._discount = discount
        self._target_update_period = target_update_period
        self._huber_loss_parameter = huber_loss_parameter
        if max_gradient_norm is None:
            max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
        self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        self._checkpoint_interval = checkpoint_interval
        self._save_directory = save_directory
        self._alpha = tf.constant(alpha, dtype=tf.float32)

        # Learner state.
        self._variables: List[List[tf.Tensor]] = [network.trainable_variables]
        self._num_steps = tf.Variable(0, dtype=tf.int32)

        # Internalise logging/counting objects.
        self._counter = counter or counting.Counter()
        self._counter.increment(learner_steps=0)

        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

        # Create a checkpointer object.
        if checkpoint:
            self._checkpointer = tf2_savers.Saver(self.state)
        else:
            self._checkpointer = None

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        """Do a step of SGD."""

        inputs = next(self._iterator)
        transitions: types.Transition = inputs.data

        with tf.GradientTape() as tape:
            # Evaluate our networks.
            q_tm1 = self._network(transitions.observation)
            q_t_value = self._target_network(transitions.next_observation)
            q_t_selector = self._network(transitions.next_observation)

            # The rewards and discounts have to have the same type as network values.
            r_t = tf.cast(transitions.reward, q_tm1.dtype)
            r_t = tf.clip_by_value(r_t, -1., 1.)
            d_t = tf.cast(transitions.discount, q_tm1.dtype) * tf.cast(
                self._discount, q_tm1.dtype)

            # Compute the loss.
            _, extra = trfl.double_qlearning(q_tm1, transitions.action, r_t, d_t,
                                            q_t_value, q_t_selector)
            loss = losses.huber(extra.td_error, self._huber_loss_parameter)

            # Compute CQL reguralizer loss.
            push_up = trfl.indexing_ops.batched_index(q_tm1, transitions.action) # Q-value under behavioural policy.
            push_down = tf.reduce_logsumexp(q_tm1, axis=1) # soft-maximum of the Q-function.

            # Compute global loss.
            cql_loss = loss + self._alpha * (push_down - push_up)
            cql_loss = tf.reduce_mean(cql_loss, axis=0)

        # Do a step of SGD.
        gradients = tape.gradient(cql_loss, self._network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._optimizer.apply(gradients, self._network.trainable_variables)

        # Periodically update the target network.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(self._network.variables,
                                self._target_network.variables):
                dest.assign(src)
        self._num_steps.assign_add(1)

        # Report loss & statistics for logging.
        fetches = {
            'critic_loss': tf.reduce_mean(loss, axis=0),
            'q_variance': tf.reduce_mean(tf.math.reduce_variance(q_tm1, axis=1), axis=0),
            'q_average': tf.reduce_mean(q_tm1),
            'push_up': tf.reduce_mean(push_up, axis=0),
            'push_down': tf.reduce_mean(push_down, axis=0),
            'regularizer': tf.reduce_mean(push_down - push_up, axis=0),
            'cql_loss': cql_loss,
        }
        return fetches

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
        if self._checkpointer is not None and \
            self._num_steps % self._checkpoint_interval == 0:
            print('Saving checkpoint.')
            self._checkpointer.save(self._save_directory + f"/chkpt_{self._num_steps.numpy()}")
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