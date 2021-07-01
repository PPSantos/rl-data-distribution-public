"""DQN2BE learner implementation."""

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


class DQN2BEUnprioritizedLearner(acme.Learner, tf2_savers.TFSaveable):
  """DQN2BE unprioritized learner.

  This is the learning component of a DQN2BE agent. It takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      bellman_error_network: snt.Module,
      target_bellman_error_network: snt.Module,
      discount: float,
      learning_rate: float,
      be_net_learning_rate: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      max_gradient_norm: float = None,
  ):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized)
      target_network: the target Q critic (which lags behind the online net).
      bellman_error_network:
      target_bellman_error_network:
      discount: discount to use for TD updates.
      learning_rate: learning rate for the q-network update.
      be_net_learning_rate: learning rate for the bellman error network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
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
    self._bellman_error_network = bellman_error_network
    self._target_bellman_error_network = target_bellman_error_network
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._belman_error_net_optimizer = snt.optimizers.Adam(be_net_learning_rate)

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
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

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:

    data, info = next(self._iterator)

    # Unpack data.
    observation = data[0]
    action = data[1]
    reward = data[2]
    discount = data[3]
    next_observation = data[4]

    with tf.GradientTape(persistent=True) as tape:

      """
        DQN loss.
      """
      # Evaluate our networks.
      q_tm1 = self._network(observation)
      q_t_value = self._target_network(next_observation)
      q_t_selector = self._network(next_observation)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(reward, q_tm1.dtype)
      r_t = tf.clip_by_value(r_t, -1., 1.)
      d_t = tf.cast(tf.ones_like(discount), q_tm1.dtype) * tf.cast(
          self._discount, q_tm1.dtype)
      # d_t = tf.cast(discount, q_tm1.dtype) * tf.cast(
      #    self._discount, q_tm1.dtype) # discount = 0 if last timestep.

      # Compute the loss.
      squared_loss, extra = trfl.double_qlearning(q_tm1, action, r_t, d_t,
                                       q_t_value, q_t_selector)
      loss = losses.huber(extra.td_error, self._huber_loss_parameter)
      loss = tf.reduce_mean(loss, axis=[0])  # []

      """
        Bellman error network loss.
      """
      be_tm1 = self._bellman_error_network(observation)
      be_t_value = self._target_bellman_error_network(next_observation)
      be_t_selector = self._bellman_error_network(next_observation)
      be_loss, _ = trfl.double_qlearning(be_tm1, action, squared_loss, d_t,
                                      be_t_value, be_t_selector)
      be_loss = tf.reduce_mean(be_loss, axis=[0])  # []

    # Do a step of SGD (DQN network).
    gradients = tape.gradient(loss, self._network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    # Do a step of SGD (Bellman error network).
    gradients = tape.gradient(be_loss, self._bellman_error_network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._belman_error_net_optimizer.apply(gradients, self._bellman_error_network.trainable_variables)

    del tape

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:

      # DQN network.
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)

      # Bellman error network.
      for src, dest in zip(self._bellman_error_network.variables,
                           self._target_bellman_error_network.variables):
        dest.assign(src)

    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
        'loss': loss,
        'be_loss': be_loss,
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
        'bellman_error_network': self._bellman_error_network,
        'target_bellman_error_network': self._target_bellman_error_network,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }