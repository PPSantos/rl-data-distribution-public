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

def boltzmann(x, t=10):
    return np.exp(x*t)/np.sum(np.exp(x*t))

def build_boltzmann_policy(qvals, temperature):
    def policy(s):
        optimal_qvals = qvals # [S,A]
        action_probs = boltzmann(optimal_qvals[s], t=temperature)
        return np.random.choice(np.arange(optimal_qvals.shape[-1]), p=action_probs)
    return policy

def build_eps_greedy_policy(qvals, epsilon):
    def policy(s):
        optimal_qvals = qvals # [S,A]
        return choice_eps_greedy(optimal_qvals[s], epsilon=epsilon)
    return policy
