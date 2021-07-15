"""DQN_E_tab agent implementation."""

import copy
from typing import Optional

from acme import specs
from acme import types
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import sonnet as snt
import tensorflow as tf
import trfl

from tf_agents.specs import tensor_spec

from algos import agent_acme, actors
from algos.tf_uniform_replay_buffer import TFUniformReplayBuffer
from algos.utils import tf2_savers, spec_converter
from algos.utils.tf2_layers import EpsilonGreedyExploration, ConvexCombination
from algos.dqn_e_tab.dqn_e_tab_acme_learning import DQN_E_tab_Learner
from algos.tf_adder import TFAdder


class DQN_E_tab(agent_acme.Agent):
    """DQN_E_tab agent.
    This implements a single-process DQN_E_tab agent. This is a simple Q-learning
    algorithm that inserts N-step transitions into a replay buffer, and
    periodically updates its policy by sampling these transitions.
    """

    def __init__(
            self,
            environment_spec: specs.EnvironmentSpec,
            network: snt.Module,
            batch_size: int = 256,
            target_update_period: int = 100,
            samples_per_insert: float = 32.0,
            min_replay_size: int = 20,
            max_replay_size: int = 1000000,
            epsilon_init: float = 1.0,
            epsilon_final: float = 0.01,
            epsilon_schedule_timesteps: int = 20_000,
            learning_rate: float = 1e-3,
            discount: float = 0.99,
            max_gradient_norm: Optional[float] = None,
            logger: loggers.Logger = None,
            num_states: int = None,
            num_actions: int = None,
        ):
        """Initialize the agent.
        Args:
        environment_spec: description of the actions, observations, etc.
        network: the online Q network (the one being optimized).
        batch_size: batch size for updates.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        min_replay_size: minimum replay size before updating. This and all
            following arguments are related to dataset construction and will be
            ignored if a dataset argument is passed.
        max_replay_size: maximum replay size.
        epsilon_init: Initial epsilon value (probability of taking a random action)
        epsilon_final: Final epsilon value (probability of taking a random action)
        epsilon_schedule_timesteps: timesteps to decay epsilon from 'epsilon_init'
            to 'epsilon_final'.
        learning_rate: learning rate for the q-network update.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.

        """
        self.num_states = num_states
        self.num_actions = num_actions

        # Create replay buffer.
        env_state_spec = tensor_spec.TensorSpec((),
                                dtype=tf.int32,
                                name='env_state')
        env_next_state_spec = tensor_spec.TensorSpec((),
                                dtype=tf.int32,
                                name='env_next_state')
        extras = (env_state_spec, env_next_state_spec)
        transition_spec = spec_converter.convert_env_spec(environment_spec, extras=extras)
        self.replay_buffer = TFUniformReplayBuffer(data_spec=transition_spec,
                                                    batch_size=1,
                                                    max_length=max_replay_size,
                                                    statistics_table_shape=(self.num_states,
                                                                            self.num_actions))
        dataset = self.replay_buffer.as_dataset(sample_batch_size=batch_size)
        self._dataset_iterator = iter(dataset)

        self.adder = TFAdder(self.replay_buffer, transition_spec)

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
        actor = actors.FeedForwardActor(policy_network, self.adder)

        # The learner updates the parameters (and initializes them).
        learner = DQN_E_tab_Learner(
            network=network,
            target_network=target_network,
            discount=discount,
            learning_rate=learning_rate,
            target_update_period=target_update_period,
            dataset=dataset,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            checkpoint=False)

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
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert)

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

    def add_to_replay_buffer(self, transition, extras=None):
        self.adder.add_op(transition, extras)

    def sample_replay_buffer_batch(self):        
        data, info = next(self._dataset_iterator)
        return [d.numpy() for d in data]
