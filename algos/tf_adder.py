from acme import types
from acme.adders.base import Adder

import dm_env

import tensorflow as tf


class TFAdder(Adder):

    def __init__(self, replay_buffer, data_spec):
        self._replay_buffer = replay_buffer
        self._data_spec = data_spec

        self._prev_timestep = None

    def add_first(self, timestep: dm_env.TimeStep):
        self._prev_timestep = timestep

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = ()):

        prev_observation = tf.constant(self._prev_timestep.observation, dtype=self._data_spec[0].dtype)
        action = tf.constant(action, dtype=self._data_spec[1].dtype)
        reward = tf.constant(next_timestep.reward, dtype=self._data_spec[2].dtype)
        discount = tf.constant(next_timestep.discount, dtype=self._data_spec[3].dtype)
        observation = tf.constant(next_timestep.observation, dtype=self._data_spec[4].dtype)

        values = (prev_observation, action, reward, discount, observation)

        # Add extras.
        for (extra, extra_spec) in zip(extras, self._data_spec[5:]):
            values += (tf.constant(extra, dtype=extra_spec.dtype),)

        transition = tf.nest.map_structure(lambda t: tf.stack([t]),
                                        values)

        self._replay_buffer.add_batch(transition)

    def reset(self):
        self._prev_timestep = None