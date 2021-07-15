"""
    Helper array functions.
"""
import numpy as np

def all_eq(values):
    # Returns True if every element of 'values' is the same.
    return all(np.isnan(values)) or max(values) - min(values) < 1e-6

def choice_eps_greedy(values, epsilon):
    if np.random.rand() <= epsilon or all_eq(values):
        return np.random.choice(len(values))
    else:
        return np.argmax(values)
