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

        transition = (self._prev_timestep.observation, action,
                next_timestep.reward, next_timestep.discount, next_timestep.observation)
        self.add_op(transition, extras)

        self._prev_timestep = next_timestep

    def add_op(self, transition, extras):

        to_add = ()
        for (item, item_spec) in zip(transition, self._data_spec[:5]):
            to_add += (tf.constant(item, dtype=item_spec.dtype),)

        for (extra, extra_spec) in zip(extras, self._data_spec[5:]):
            to_add += (tf.constant(extra, dtype=extra_spec.dtype),)

        to_add = tf.nest.map_structure(lambda t: tf.stack([t]),
                                            to_add)

        self._replay_buffer.add_batch(to_add)

    def reset(self):
        self._prev_timestep = None
