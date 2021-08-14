import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from acme import types
from typing import Callable, Optional, Union
TensorTransformation = Union[snt.Module, Callable[[types.NestedTensor],
                                                tf.Tensor]]

class GaussianNoiseExploration(snt.Module):
    """ Sonnet module for adding gaussian noise (exploration). """

    def __init__(self,
                 eval_mode: bool,
                 stddev_init: float = 0.4,
                 stddev_final: float = 0.01,
                 stddev_schedule_timesteps: int = 25000,
                 ):
        """ Initialise GaussianNoise class.
            Parameters:
            ----------
            * eval_mode: bool
                If eval_mode is True then this module does not affect
                input values.
            * stddev_init: int
                Initial stddev value.
            * stddev_final: int
                Final stddev value.
            * stddev_schedule_timesteps: int
                Number of timesteps to decay stddev from 'stddev_init'
                to 'stddev_final'
        """
        super().__init__(name='gaussian_noise_exploration')

        # Internalise parameters.
        self._stddev_init = stddev_init
        self._stddev_final = stddev_final
        self._stddev_schedule_timesteps = stddev_schedule_timesteps
        self._eval_mode = tf.Variable(eval_mode)

        # Internal counter.
        self._counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        # Calculate new stddev value.
        self._counter.assign(self._counter + 1.0)
        fraction = tf.math.minimum(self._counter / self._stddev_schedule_timesteps, 1.0)
        stddev = self._stddev_init + fraction * (self._stddev_final - self._stddev_init)

        # Add noise. If eval_mode is True then no noise is added. If
        # eval_mode is False (training mode) then gaussian noise is added to the inputs.
        noise = tf.where(self._eval_mode,
                        tf.zeros_like(inputs),
                        tfp.distributions.Normal(loc=0., scale=stddev).sample(inputs.shape))

        # Add noise to inputs.
        output = inputs + noise

        return output


class EpsilonGreedyExploration(snt.Module):
    """ Sonnet module for epsilon-greedy exploration. """

    def __init__(self,
                 epsilon_init: float,
                 epsilon_final: float,
                 epsilon_schedule_timesteps: int):
        """ Initialise EpsilonGreedyExploration class.
            Parameters:
            ----------
            * epsilon_init: float
                Initial epsilon value.
            * epsilon_final: float
                Final epsilon value.
            * epsilon_schedule_timesteps: int
                Number of timesteps to decay epsilon from 'epsilon_init'
                to 'epsilon_final'
        """
        super().__init__(name='epsilon_greedy_exploration')

        # Internalise parameters.
        self._epsilon_init = epsilon_init
        self._epsilon_final = epsilon_final
        self._epsilon_schedule_timesteps = epsilon_schedule_timesteps

        # Internal counter.
        self._counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        num_actions = tf.cast(tf.shape(inputs)[-1], inputs.dtype)

        # Dithering action distribution.
        dither_probs = 1 / num_actions * tf.ones_like(inputs)

        # Greedy action distribution, breaking ties uniformly at random.
        max_value = tf.reduce_max(inputs, axis=-1, keepdims=True)
        greedy_probs = tf.cast(tf.equal(inputs, max_value),
                            inputs.dtype)
        greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

        # Calculate new epsilon value.
        self._counter.assign(self._counter + 1.0)
        fraction = tf.math.minimum(self._counter / self._epsilon_schedule_timesteps, 1.0)
        epsilon = self._epsilon_init + fraction * (self._epsilon_final - self._epsilon_init)

        # Epsilon-greedy action distribution.
        probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

        # Construct the policy.
        policy = tfp.distributions.Categorical(probs=probs)

        # Sample from policy.
        sample = policy.sample()

        return sample

@snt.allow_empty_variables
class InputStandardization(snt.Module):
    """ Sonnet module to scale inputs. """

    def __init__(self, shape):
        """ Initialise InputStandardization class.
            Parameters:
            ----------
            * shape: state space shape.
        """
        super().__init__(name='normalization')

        # Internalise parameters.
        self._mean = tf.Variable(tf.zeros(shape=shape), trainable=False)
        self._var = tf.Variable(tf.ones(shape=shape), trainable=False)
        self._count = tf.Variable(1e-4, trainable=False)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])
        batch_count = tf.cast(tf.shape(inputs)[0], inputs.dtype)

        if batch_count > 1:
            # Update moving average and std.
            delta = batch_mean - self._mean
            tot_count = self._count + batch_count

            self._mean.assign(self._mean + delta * batch_count / tot_count)
            m_a = self._var * self._count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + tf.math.square(delta) * self._count * batch_count / tot_count
            self._var.assign(M2 / tot_count)
            self._count.assign(tot_count)

        # Standardize inputs.
        normalized = (inputs - self._mean) / self._var

        return normalized

class ConvexCombination(snt.Module):
    """ Sonnet module to linearly combine two inputs. """

    def __init__(self,
                 network_1: TensorTransformation,
                 network_2: TensorTransformation,
                 delta_init: int,
                 delta_final: int,
                 delta_schedule_timesteps: int):
        """ Initialise ConvexCombination class.
            Parameters:
            ----------
            * network_1
                first input network.
            * network_2
                second input network.
            * delta_init: int
                Initial delta value.
            * delta_final: int
                Final delta value.
            * delta_schedule_timesteps: int
                Number of timesteps to decay delta from 'delta_init'
                to 'delta_final'
        """
        super().__init__(name='convex_combination')

        self._network_1 = network_1
        self._network_2 = network_2

        # Internalise parameters.
        self._delta_init = delta_init
        self._delta_final = delta_final
        self._delta_schedule_timesteps = delta_schedule_timesteps

        # Internal counter.
        self._counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        out_1 = self._network_1(inputs)
        out_2 = self._network_2(inputs)

        # Calculate new delta value.
        self._counter.assign(self._counter + 1.0)
        fraction = tf.math.minimum(self._counter / self._delta_schedule_timesteps, 1.0)
        delta = self._delta_init + fraction * (self._delta_final - self._delta_init)

        out = (1 - delta) * out_1 + delta * out_2

        return out

class CustomExplorationNet(snt.Module):

    def __init__(self,
                 q_network: TensorTransformation,
                 e_network: TensorTransformation,
                 delta: float,
                 epsilon_init: float,
                 epsilon_final: float,
                 epsilon_schedule_timesteps: int):
        super().__init__(name='custom_exploration_net')

        self._q_network = q_network
        self._e_network = e_network

        # Internalise parameters.
        self._delta = delta
        self._epsilon_init = epsilon_init
        self._epsilon_final = epsilon_final
        self._epsilon_schedule_timesteps = epsilon_schedule_timesteps

        # Internal counter.
        self._counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        q_vals = self._q_network(inputs)
        e_vals = self._e_network(inputs)

        # Greedy action distribution w.r.t. Q-values, breaking ties uniformly at random.
        max_value_q = tf.reduce_max(q_vals, axis=-1, keepdims=True)
        greedy_probs_q = tf.cast(tf.equal(q_vals, max_value_q),
                            q_vals.dtype)
        greedy_probs_q /= tf.reduce_sum(greedy_probs_q, axis=-1, keepdims=True)

        # Greedy action distribution w.r.t. E-values, breaking ties uniformly at random.
        max_value_e = tf.reduce_max(e_vals, axis=-1, keepdims=True)
        greedy_probs_e = tf.cast(tf.equal(e_vals, max_value_e),
                            e_vals.dtype)
        greedy_probs_e /= tf.reduce_sum(greedy_probs_e, axis=-1, keepdims=True)

        # Weight distribution by delta coefficient.
        net_probs = self._delta * greedy_probs_e + (1 - self._delta) * greedy_probs_q

        # Dithering action distribution.
        num_actions = tf.cast(tf.shape(q_vals)[-1], q_vals.dtype)
        dither_probs = 1 / num_actions * tf.ones_like(q_vals)

        # Calculate new epsilon value.
        self._counter.assign(self._counter + 1.0)
        fraction = tf.math.minimum(self._counter / self._epsilon_schedule_timesteps, 1.0)
        epsilon = self._epsilon_init + fraction * (self._epsilon_final - self._epsilon_init)

        # Epsilon-greedy action distribution.
        probs = epsilon * dither_probs + (1 - epsilon) * net_probs

        # Construct the policy.
        policy = tfp.distributions.Categorical(probs=probs)

        # Sample from policy.
        sample = policy.sample()

        return sample
