import numpy as np

class Discretizer(object):
    def __init__(self, dim_bins=10):
        super(Discretizer, self).__init__()

        self.ENV_NUM_DIMS = 2
        self.ENV_OBSERVATION_SPACE_LOW = [-5.0, -5.0]
        self.ENV_OBSERVATION_SPACE_HIGH = [5.0, 5.0]

        self.dim_bins = dim_bins
        self.num_states = dim_bins**self.ENV_NUM_DIMS
        self.state_dims = self.ENV_NUM_DIMS

        self.bins = []
        self.bins_size = []
        for s_dim in range(self.state_dims):
            self.bins.append(np.linspace(self.ENV_OBSERVATION_SPACE_LOW[s_dim],
                                    self.ENV_OBSERVATION_SPACE_HIGH[s_dim],
                                    dim_bins+1))
            self.bins_size.append((self.ENV_OBSERVATION_SPACE_HIGH[s_dim] - \
                                self.ENV_OBSERVATION_SPACE_LOW[s_dim]) / dim_bins)

        print(self.bins)
        print(self.bins_size)

    def set_state(self, state):
        print('set_state()')
        print('state:', state)
        continuous_state = ()
        remainder = state
        for s_dim in range(self.state_dims):
            dim_state = remainder // self.dim_bins**(self.state_dims - 1 - s_dim)
            remainder -= dim_state * self.dim_bins**(self.state_dims - 1 - s_dim)
            dim_continuous_state = (self.ENV_OBSERVATION_SPACE_LOW[s_dim] + \
                dim_state*self.bins_size[s_dim]) + np.random.rand()*self.bins_size[s_dim]
            continuous_state += (dim_continuous_state,)

        print('continuous_state:', continuous_state)

    def get_state(self, continuous_state):
        print('get_state()')
        print('continuous_state', continuous_state)
        discretized_state = 0
        for s_dim in range(self.state_dims):
            trimmed_dim_state_value = np.minimum(continuous_state[s_dim], \
                            self.ENV_OBSERVATION_SPACE_HIGH[s_dim] - 1e-05)
            s_dim_idx = np.digitize(trimmed_dim_state_value, bins=self.bins[s_dim]) - 1
            discretized_state += s_dim_idx * self.dim_bins**(self.state_dims - 1 - s_dim)

        print('discretized_state:', discretized_state)
        return discretized_state


if __name__ == "__main__":

    discretizer = Discretizer()

    discretizer.get_state((0.5, 0.5))
    discretizer.get_state((0.5, 1.9))
    discretizer.get_state((1.9, 9.99))

    discretizer.set_state(0)
    discretizer.set_state(5)
    discretizer.set_state(5)
    discretizer.set_state(17)
    discretizer.set_state(19)
    discretizer.set_state(99)

