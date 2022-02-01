"""
    Acme environment utils.
"""
from acme import wrappers

def wrap_env(env):
    return wrappers.wrap_all(env, [
        wrappers.GymWrapper,
        wrappers.SinglePrecisionWrapper,
    ])

def run_rollout(env, agent):
    cumulative_reward = 0.0
    timestep = env.reset()

    while not timestep.last():
        action = agent.deterministic_action(timestep.observation)
        timestep = env.step(action)
        cumulative_reward += timestep.reward

    return cumulative_reward
