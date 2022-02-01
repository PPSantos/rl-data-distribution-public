"""DQN agent implementation."""

import copy
from typing import Optional

from acme import types
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import reverb
import trfl
import sonnet as snt

from utils import tf2_layers, tf2_savers
from algos.dqn.learning import DQNLearner


class DQN(agent.Agent):
    """DQN agent.

    This implements a single-process DQN agent. This is a simple Q-learning
    algorithm that inserts N-step transitions into a replay buffer, and
    periodically updates its policy by sampling these transitions using
    prioritization.

    """
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network: snt.Module,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1_000,
        max_replay_size: int = 1_000_000,
        n_step: int = 5,
        epsilon_init: float = 1.0,
        epsilon_final: float = 1.0,
        epsilon_schedule_timesteps: int = 1_000_000,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        logger: Optional[loggers.Logger] = None,
        max_gradient_norm: Optional[float] = None,
        checkpoint: bool = True,
        checkpoint_interval: int = 5_000,
        save_directory: str = '~/acme',
    ):
        """Initialize the agent.

        Args:
            environment_spec: description of the actions, observations, etc.
            network: the online Q network (the one being optimized)
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_update_period: number of learner steps to perform before updating
                the target networks.
            samples_per_insert: number of samples to take from replay for every insert
                that is made.
            min_replay_size: minimum replay size before updating. This and all
                following arguments are related to dataset construction and will be
                ignored if a dataset argument is passed.
            max_replay_size: maximum replay size.
            n_step: number of steps to squash into a single transition.
            epsilon_init: Initial epsilon value (probability of taking a random action)
            epsilon_final: Final epsilon value (probability of taking a random action)
            epsilon_schedule_timesteps: timesteps to decay epsilon from 'epsilon_init'
                to 'epsilon_final'. 
            learning_rate: learning rate for the q-network update.
            discount: discount to use for TD updates.
            logger: logger object to be used by learner.
            max_gradient_norm: used for gradient clipping.
            checkpoint: whether to save checkpoint.
            checkpoint_interval.
            save_directory.
        """

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec))
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        replay_client = reverb.Client(address)
        dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=batch_size,
            prefetch_size=prefetch_size)

        # Create a epsilon-greedy policy network.
        policy_network = snt.Sequential([
            network,
            tf2_layers.EpsilonGreedyExploration(epsilon_init=epsilon_init,
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
        learner = DQNLearner(
            network=network,
            target_network=target_network,
            discount=discount,
            learning_rate=learning_rate,
            target_update_period=target_update_period,
            dataset=dataset,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_interval=checkpoint_interval,
            save_directory=save_directory,
        )

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

    # def save(self, p: str):
    #     self._saver.save(p)

    def load(self, p: str):
        self._saver.load(p)
