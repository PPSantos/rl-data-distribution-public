"""
    tf2_savers.py
"""
import pickle
from typing import Mapping, Union

import tensorflow as tf
from acme import core

PythonState = tf.train.experimental.PythonState
Checkpointable = Union[tf.Module, tf.Variable, PythonState] 


class Saver:
    """
        Convenience class to save tf.train.Checkpoints.
    """

    def __init__(
        self,
        objects_to_save: Mapping[str, Union[Checkpointable, core.Saveable]],
    ):

        # Convert `Saveable` objects to TF `Checkpointable` first, if necessary.
        def to_ckptable(x: Union[Checkpointable, core.Saveable]) -> Checkpointable:
            if isinstance(x, core.Saveable):
                return SaveableAdapter(x)
            return x

        objects_to_save = {k: to_ckptable(v) for k, v in objects_to_save.items()}

        self._checkpoint = tf.train.Checkpoint(**objects_to_save)

    def save(self, path):
        self._checkpoint.write(path)

    def load(self, path):
        self._checkpoint.read(path)


class SaveableAdapter(tf.train.experimental.PythonState):
  """Adapter which allows `Saveable` object to be checkpointed by TensorFlow."""

  def __init__(self, object_to_save: core.Saveable):
    self._object_to_save = object_to_save

  def serialize(self):
    state = self._object_to_save.save()
    return pickle.dumps(state)

  def deserialize(self, pickled: bytes):
    state = pickle.loads(pickled)
    self._object_to_save.restore(state)