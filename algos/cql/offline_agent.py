"""
    Offline CQL agent implementation.
"""
import copy
from typing import Optional

import trfl
import sonnet as snt
import tensorflow as tf
from acme import types
from acme import specs
from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from utils import tf2_savers
from algos.cql.learning import CQLLearner


class OfflineCQL(object):
    """
        Offline CQL agent.
    """
    def __init__(
        self,
        env_spec: specs.EnvironmentSpec,
        network: snt.Module,
        dataset: tf.data.Dataset,
        target_update_period: int = 100,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        alpha: float = 1e-03,
        logger: Optional[loggers.Logger] = None,
        max_gradient_norm: Optional[float] = None,
        checkpoint: bool = True,
        checkpoint_interval: int = 5_000,
        save_directory: str = '~/acme',
    ):
        """Initialize the agent.

        Args:
            env_spec: description of the actions, observations, etc.
            network: the online Q network (the one being optimized)
            dataset: dataset containing transitions.
            target_update_period: number of learner steps to perform before updating
                the target networks.
            learning_rate: learning rate for the q-network update.
            discount: discount to use for TD updates.
            alpha: CQL reguralizer coefficient.
            logger: logger object to be used by learner.
            max_gradient_norm: used for gradient clipping.
            checkpoint: whether to save checkpoint.
            checkpoint_interval: interval to save checkpoints.
            save_directory: checkpoints directory.
        """

        # Create target network and initialize networks.
        target_network = copy.deepcopy(network)
        tf2_utils.create_variables(network, [env_spec.observations])
        tf2_utils.create_variables(target_network, [env_spec.observations])

        # Create actors.
        greedy_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=0.0).sample(),
        ])
        self._greedy_actor = actors.FeedForwardActor(greedy_network)
        self._q_vals_actor = actors.FeedForwardActor(network)

        # Create learner.
        self._learner = CQLLearner(
            network=network,
            target_network=target_network,
            discount=discount,
            learning_rate=learning_rate,
            alpha=alpha,
            target_update_period=target_update_period,
            dataset=dataset,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_interval=checkpoint_interval,
            save_directory=save_directory,
        )

        self._saver = tf2_savers.Saver(self._learner.state)

    def step(self):
        self._learner.step()

    def get_Q_vals(self, obs: types.NestedArray) -> types.NestedArray:
        return self._q_vals_actor.select_action(obs)

    def deterministic_action(self, obs: types.NestedArray) -> types.NestedArray:
        return self._greedy_actor.select_action(obs)

    def save(self, p: str):
        self._saver.save(p)

    def load(self, p: str):
        self._saver.restore(p)
