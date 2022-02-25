import numpy as np
from scipy import stats

DIST_SIZE = 20*20*20*20*2

uniform_dist = np.ones(DIST_SIZE) / DIST_SIZE
print(uniform_dist)

entropy = stats.entropy(uniform_dist)
print(entropy)