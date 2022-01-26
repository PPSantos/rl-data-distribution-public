import numpy as np

def get_env(env):

    class DiscretizableEnv(env):
        def __init__(self, dim_bins=20):
            super(DiscretizableEnv, self).__init__()

            self.dim_bins = dim_bins
            self.num_states = dim_bins**self.observation_space.shape[0]
            self.num_actions = self.action_space.n
            self.state_dims = self.observation_space.shape[0]

            self.bins = []
            self.bins_size = []
            for s_dim in range(self.state_dims):
                self.bins.append(np.linspace(self.observation_space.low[s_dim],
                                        self.observation_space.high[s_dim],
                                        dim_bins+1))
                self.bins_size.append((self.observation_space.high[s_dim] - \
                                    self.observation_space.low[s_dim]) / dim_bins)

            #print(self.bins)
            #print(self.bins_size)

        def _get_continuous_state(self, state):
            continuous_state = ()
            remainder = state
            for s_dim in range(self.state_dims):
                dim_state = remainder // self.dim_bins**(self.state_dims - 1 - s_dim)
                remainder -= dim_state * self.dim_bins**(self.state_dims - 1 - s_dim)
                dim_continuous_state = (self.observation_space.low[s_dim] + \
                    dim_state*self.bins_size[s_dim]) + np.random.rand()*self.bins_size[s_dim]
                continuous_state += (dim_continuous_state,)
            return continuous_state

        def get_observation(self, state):
            return np.array(self._get_continuous_state(state))

        def set_state(self, state):
            self.state = self._get_continuous_state(state)

        def get_state(self):
            state = self.state
            discretized_state = 0
            for s_dim in range(self.state_dims):
                trimmed_dim_state_value = np.minimum(state[s_dim], \
                                self.observation_space.high[s_dim] - 1e-05)
                s_dim_idx = np.digitize(trimmed_dim_state_value, bins=self.bins[s_dim]) - 1
                discretized_state += s_dim_idx * self.dim_bins**(self.state_dims - 1 - s_dim)
            return discretized_state

    return DiscretizableEnv
