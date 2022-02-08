import sys
import gym
import numpy as np
from envs import grid_spec

class MultiPathEnv(gym.Env):
    def __init__(self, max_timesteps=20):

        self.max_timesteps = max_timesteps
        self.num_states = 5*5+2
        self.num_actions = 5

        np.random.seed(33)

        self._timestep = 0

        self._correct_actions = np.random.randint(low=0, high=5, size=5*5+2, dtype=np.int32)
        print('self._correct_actions', self._correct_actions)

        self._init_action_random_p = 0.01 # first action randomness.

        super(MultiPathEnv, self).__init__()

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.num_actions)

    @property
    def observation_space(self):
        return gym.spaces.Discrete(self.num_states)

    def get_state(self):
        return self.__state

    def set_state(self, s):
        self.__state = s

    def _get_transitions(self, s, a):

        if s == 0: # Start state.
            probs = [self._init_action_random_p]*5
            probs[a] += 1 - self._init_action_random_p*5
            return [1,6,11,16,21], probs

        elif s == 26: # Terminal (dead) state.
            return [s], [1.0]

        elif s in [5,10,15,20,25]: # Terminal (win) state.
            return [s], [1.0]

        else:
            good_action_idx = self._correct_actions[s]
            if a == good_action_idx:
                return [s+1], [1.0] # Move one step forward.
            else:
                return [26], [1.0] # Move to dead state.

    def get_transitions(self, s, a):
        next_states, probs = self._get_transitions(s, a)
        return {next_state: prob for next_state, prob in zip(next_states, probs)}

    def get_reward(self, state):
        if state in [5,10,15,20,25]:
            return 1.0
        else:
            return 0.0

    def step_stateless(self, s, a, verbose=False):
        next_states, probs = self._get_transitions(s, a)
        next_s = np.random.choice(next_states, p=probs)
        r = self.get_reward(s)

        return next_s, r

    def step(self, a):
        ns, r = self.step_stateless(self.__state, a)
        self.__state = ns
        done = False
        info = {}
        self._timestep += 1
        if self._timestep >= self.max_timesteps:
            info["TimeLimit.truncated"] = True
            done = True
        return ns, r, done, info
        
    def reset(self):
        start_state = 0
        self.__state = 0
        self._timestep = 0
        return start_state

    def render(self, close=False, ostream=sys.stdout):
        print(self.__state)


class MultiPathRandomObservation(MultiPathEnv):

    def __init__(self, features_dim=8, **kwargs):

        super(MultiPathRandomObservation, self).__init__(**kwargs)

        self.obs_dim = features_dim + 4
        self._features = np.random.random((self.num_states, features_dim)).astype(np.float32)*2 - 1

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0,
                            shape=(self.obs_dim,), dtype=np.float32)

    def _get_observation(self, state):
        obs = self._features[state]
        if state == 0:
            return np.append(obs, [1,0,0,0]).astype(np.float32)
        elif state == 26:
            return np.append(obs, [0,1,0,0]).astype(np.float32)
        elif state in [5,10,15,20,25]:
            return np.append(obs, [0,0,1,0]).astype(np.float32)
        else:
            return np.append(obs, [0,0,0,1]).astype(np.float32)
    
    def get_observation(self, state):
        return self._get_observation(state)

    def step(self, a):
        next_s, r, done, info = super().step(a)
        return self._get_observation(next_s), r, done, info

    def reset(self):
        state = super().reset()
        return self._get_observation(state)
