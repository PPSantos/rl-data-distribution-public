import time
from typing import Dict, List


import acme
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl


class DQN_E_func_Learner(acme.Learner, tf2_savers.TFSaveable):
  """
    DQN_E_func_Learner.
  """

  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      e_network: snt.Module,
      target_e_network: snt.Module,
      discount: float,
      learning_rate: float,
      e_net_learning_rate: float,
      target_update_period: int,
      target_e_net_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      max_gradient_norm: float = None,
  ):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized).
      target_network: the target Q critic (which lags behind the online net).
      e_network: the online E-values network (the one being optimized).
      target_e_network: the target E-values critic (which lags behind the online net).
      discount: discount to use for TD updates.
      learning_rate: learning rate for the q-network update.
      e_net_learning_rate: learning rate for the E-values network update.
      target_update_period: number of learner steps to perform before updating
        the Q-values target network.
      target_e_net_update_period: number of learner steps to perform before updating
        the E-values target network.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner.
      max_gradient_norm: used for gradient clipping.
    """
    # Internalise agent components (replay buffer, networks, optimizer).
    self._iterator = iter(dataset)
    self._network = network
    self._target_network = target_network
    self._e_network = e_network
    self._target_e_network = target_e_network
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._e_net_optimizer = snt.optimizers.Adam(e_net_learning_rate)

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
    self._target_e_net_update_period = target_e_net_update_period
    self._huber_loss_parameter = huber_loss_parameter
    if max_gradient_norm is None:
      max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
    self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

    # Learner state.
    self._variables: List[List[tf.Tensor]] = [network.trainable_variables,
                                              e_network.trainable_variables]
    self._num_steps_q = tf.Variable(0, dtype=tf.int32)
    self._num_steps_e = tf.Variable(0, dtype=tf.int32)

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
  def _step_q(self) -> Dict[str, tf.Tensor]:

    data, info = next(self._iterator)

    # Unpack data.
    observation, action, reward, discount, next_observation, state, _, _ = data

    with tf.GradientTape() as tape:
      # Evaluate Q networks.
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
      _, extra = trfl.double_qlearning(q_tm1, action, r_t, d_t,
                                       q_t_value, q_t_selector)
      loss = losses.huber(extra.td_error, self._huber_loss_parameter)
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD (Q network).
    gradients = tape.gradient(loss, self._network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    # Periodically update the Q target network.
    if tf.math.mod(self._num_steps_q, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)

    self._num_steps_q.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
        'q_loss': loss,
    }

    return fetches

  @tf.function
  def _step_e(self) -> Dict[str, tf.Tensor]:

    data, info = next(self._iterator)

    # Unpack data.
    observation, action, reward, discount, next_observation, _, _, targets = data

    """
    # Oracle version.
    q_tm1 = self._network(observation) # [B,A]
    qa_tm1 = trfl.indexing_ops.batched_index(q_tm1, action) # [B]
    error = targets - qa_tm1 # [B]
    q_loss = losses.huber(error, self._huber_loss_parameter)
    # Rescale and clip Q-loss.
    #q_loss = 0.01 * q_loss
    #q_loss = tf.clip_by_value(q_loss, -1., 1.) """

    # TD-error version.
    q_tm1 = self._network(observation) # [B,A]
    q_t = self._network(next_observation) # [B,A]
    r_t = tf.cast(reward, q_tm1.dtype)
    r_t = tf.clip_by_value(r_t, -1., 1.)
    d_t = tf.cast(tf.ones_like(discount), q_tm1.dtype) * tf.cast(
        self._discount, q_tm1.dtype)
    _, extra = trfl.qlearning(q_tm1, action, r_t, d_t, q_t)
    q_loss = losses.huber(extra.td_error, self._huber_loss_parameter) # [B]

    with tf.GradientTape() as tape:
      # Evaluate E networks.
      e_tm1 = self._e_network(observation)
      e_t_value = self._target_e_network(next_observation)
      e_t_selector = self._e_network(next_observation)
      d_t = tf.cast(tf.ones_like(discount), e_tm1.dtype) * tf.cast(
          self._discount, e_tm1.dtype)
      _, extra = trfl.double_qlearning(e_tm1, action, q_loss, d_t,
                                      e_t_value, e_t_selector)
      e_loss = losses.huber(extra.td_error, self._huber_loss_parameter)
      e_loss = tf.reduce_mean(e_loss, axis=[0])  # []

    # Do a step of SGD (E network).
    gradients = tape.gradient(e_loss, self._e_network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._e_net_optimizer.apply(gradients, self._e_network.trainable_variables)

    # Periodically update the E target network.
    if tf.math.mod(self._num_steps_e, self._target_e_net_update_period) == 0:

      for src, dest in zip(self._e_network.variables,
                           self._target_e_network.variables):
        dest.assign(src)

    self._num_steps_e.assign_add(1)

    return {'e_loss': e_loss, 'max_q_loss': tf.reduce_max(q_loss)}

  def step(self):
    # Do a batch of SGD.
    result = {}
    result.update(self._step_q())
    result.update(self._step_e())

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
        'e_network': self._e_network,
        'target_e_network': self._target_e_network,
        'optimizer': self._optimizer,
        'num_steps_q': self._num_steps_q,
        'num_steps_e': self._num_steps_e,
    }