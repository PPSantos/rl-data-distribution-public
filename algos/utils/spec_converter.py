import tensorflow as tf

from tf_agents.specs import tensor_spec


# TODO: Make this creation dynamic.
def convert_env_spec(environment_spec, extras=None):

    spec = (
            tensor_spec.BoundedTensorSpec(environment_spec.observations.shape,
                                dtype=tf.float32,
                                minimum=environment_spec.observations.minimum,
                                maximum=environment_spec.observations.maximum,
                                name='observation'),
            tensor_spec.TensorSpec(environment_spec.actions.shape,
                                dtype=tf.int32,
                                name='action'),
            tensor_spec.TensorSpec(environment_spec.rewards.shape,
                                dtype=tf.float32,
                                name='reward'),
            tensor_spec.TensorSpec(environment_spec.discounts.shape,
                                dtype=tf.float32,
                                name='discount'),
            tensor_spec.BoundedTensorSpec(environment_spec.observations.shape,
                                dtype=tf.float32,
                                minimum=environment_spec.observations.minimum,
                                maximum=environment_spec.observations.maximum,
                                name='next_observation'),
    )

    if extras is not None:
        spec += extras

    return spec 
