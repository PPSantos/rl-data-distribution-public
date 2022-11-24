import sys
import gym
import numpy as np
from envs import grid_spec


class FourStateMDP(gym.Env):

    def __init__(self, max_timesteps=5):

        self.max_timesteps = max_timesteps
        self.num_states = 4
        self.num_actions = 2

        np.random.seed(33)

        self._timestep = 0

        self._reward_matrix = np.array([[1.0,-0.1],[-0.35,0.3],[0.0,0.0],[0.0,0.0]])

        self._transition_matrix = np.array([ [[0.0, 0.01, 0.99, 0.0], [0.0, 1.0, 0.0, 0.0]],
                                             [[0.0, 0.0,  0.0,  1.0], [0.0, 0.0, 1.0, 0.0]],
                                             [[0.0, 0.0,  1.0,  0.0], [0.0, 0.0, 1.0, 0.0]],
                                             [[0.0, 0.0,  0.0,  1.0], [0.0, 0.0, 0.0, 1.0]],
                                            ])

        super(FourStateMDP, self).__init__()

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
        next_states = []
        probs = []
        for ns in range(self.num_states):
            prob = self._transition_matrix[s,a,ns]
            if prob > 0:
                next_states.append(ns)
                probs.append(prob)

        return next_states, probs

    def get_transitions(self, s, a):
        next_states, probs = self._get_transitions(s, a)
        return {next_state: prob for next_state, prob in zip(next_states, probs)}

    def get_reward(self, s, a):
        return self._reward_matrix[s, a]

    def step_stateless(self, s, a, verbose=False):
        next_states, probs = self._get_transitions(s, a)
        next_s = np.random.choice(next_states, p=probs)
        r = self.get_reward(s,a)

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

if __name__ == "__main__":
    
    actions = [1,1,1,0,0]

    env = FourStateMDP()

    s = env.reset()
    print(s)

    done = False
    t = 0
    while not done:

        s, r, done, _ = env.step(actions[t])
        print(s, r)

        t += 1
