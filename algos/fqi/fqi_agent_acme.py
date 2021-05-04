from typing import List

from acme import core
from acme import types

import dm_env
import numpy as np


class FQIAgent(core.Actor, core.VariableSource):

    def __init__(self, actor: core.Actor, learner: core.Learner,
                num_sampling_steps: int, num_gradient_steps: int):
        self._actor = actor
        self._learner = learner
        self._num_sampling_steps = num_sampling_steps
        self._num_gradient_steps = num_gradient_steps
        self._num_observations = 0

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        return self._actor.select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self._num_observations += 1
        self._actor.observe(action, next_timestep)

    def observe_with_extras(self, action: types.NestedArray, next_timestep: dm_env.TimeStep, extras=None):
        self._num_observations += 1
        self._actor.observe_with_extras(action, next_timestep, extras)
        self._learner.update_replay_buffer_counts(extras[0], action)

    def update(self):
        if self._num_observations % self._num_sampling_steps == 0:

            # Run the learner num_gradient_steps.
            for _ in range(self._num_gradient_steps):
                self._learner.step()

            # Update the weights of the target network.
            self._learner.update_target_network()

            # Update the actor weights when learner updates.
            self._actor.update()

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return self._learner.get_variables(names)
