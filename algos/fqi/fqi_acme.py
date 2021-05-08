"""FQI agent implementation."""

import copy
from typing import Optional

import numpy as np

from acme import specs
from acme import types
from algos import agent
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import sonnet as snt
import tensorflow as tf
import trfl

from tf_agents.specs import tensor_spec

from algos import actors
from algos.tf_uniform_replay_buffer import TFUniformReplayBuffer
from algos.utils import tf2_savers, spec_converter
from algos.utils.tf2_layers import EpsilonGreedyExploration
from algos.fqi.fqi_acme_learning import FQILearner
from algos.tf_adder import TFAdder


class FQI(agent.Agent):
    """
        FQI agent.
    """

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            network: snt.Module,
            batch_size: int = 256,
            prefetch_size: int = 4,
            num_sampling_steps: int = 1000,
            num_gradient_steps: int = 10,
            max_replay_size: int = 10000,
            n_step: int = 1,
            epsilon_init: float = 1.0,
            epsilon_final: float = 0.01,
            epsilon_schedule_timesteps: int = 20000,
            learning_rate: float = 1e-3,
            discount: float = 0.99,
            max_gradient_norm: Optional[float] = None,
            logger: loggers.Logger = None,
            reweighting_type: str = 'default',
            num_states: int = None,
            num_actions: int = None,
        ):
        """Initialize the agent.
        Args:
        environment_spec: description of the actions, observations, etc.
        network: the online Q network (the one being optimized).
        batch_size: batch size for updates.
        prefetch_size: size to prefetch from replay.
        num_sampling_steps: number of sampling steps to perform.
        num_gradient_steps: number of gradient descent steps to perform.
        max_replay_size: maximum size of replay buffer.
        n_step: number of steps to squash into a single transition.
        epsilon_init: Initial epsilon value (probability of taking a random action)
        epsilon_final: Final epsilon value (probability of taking a random action)
        epsilon_schedule_timesteps: timesteps to decay epsilon from 'epsilon_init'
            to 'epsilon_final'. 
        learning_rate: learning rate for the q-network update.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.
        reweighting_type: loss importance sampling reweighting type. 
        """

        self.num_states = num_states
        self.num_actions = num_actions

        # Create replay buffer.
        extras = (tensor_spec.TensorSpec((),
                                dtype=tf.int32,
                                name='env_state'),)
        transition_spec = spec_converter.convert_env_spec(environment_spec, extras=extras)
        self.replay_buffer = TFUniformReplayBuffer(data_spec=transition_spec,
                                                    batch_size=1,
                                                    max_length=max_replay_size,
                                                    statistics_table_shape=(self.num_states,
                                                                            self.num_actions))
        dataset = self.replay_buffer.as_dataset(sample_batch_size=batch_size)
        adder = TFAdder(self.replay_buffer, transition_spec)

        # Create a epsilon-greedy policy network.
        policy_network = snt.Sequential([
            network,
            EpsilonGreedyExploration(epsilon_init=epsilon_init,
                                     epsilon_final=epsilon_final,
                                     epsilon_schedule_timesteps=epsilon_schedule_timesteps)
        ])

        # Create a target network.
        target_network = copy.deepcopy(network)

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(network, [environment_spec.observations])
        tf2_utils.create_variables(target_network, [environment_spec.observations])

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(policy_network, adder)

        # The learner updates the parameters (and initializes them).
        learner = FQILearner(
            network=network,
            target_network=target_network,
            discount=discount,
            learning_rate=learning_rate,
            dataset=dataset,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            checkpoint=False,
            reweighting_type=reweighting_type)

        self._saver = tf2_savers.Saver(learner.state)

        # Deterministic (max-Q) actor.
        max_Q_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=0.0).sample(),
        ])
        self._deterministic_actor = actors.FeedForwardActor(max_Q_network)

        self._Q_vals_actor = actors.FeedForwardActor(network)

        super().__init__(
            actor=actor,
            learner=learner,
            num_sampling_steps=num_sampling_steps,
            num_gradient_steps=num_gradient_steps)

    def update(self):
        super().update()

    def get_Q_vals(self, obs: types.NestedArray) -> types.NestedArray:
        return self._Q_vals_actor.select_action(obs)

    def deterministic_action(self, obs: types.NestedArray) -> types.NestedArray:
        return self._deterministic_actor.select_action(obs)

    def save(self, p):
        self._saver.save(p)

    def load(self, p):
        self._saver.load(p)

    def get_replay_buffer_counts(self):
        print('Getting replay buffer counts...')
        return self.replay_buffer.get_statistics()
