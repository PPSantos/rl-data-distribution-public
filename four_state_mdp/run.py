"""
    Four-state MDP experiments.
"""
import os
import json
import pathlib

from algos.linear_approximator import LinearApproximator
from four_state_mdp.env import FourStateMDP

from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

###########################################################################
ARGS = {
    "env_name": "four_state_mdp",
    "algo": "linear_approximator",

    "num_episodes": 5_000, # Number of episodes to train.
    "q_vals_period": 100, # Period at which Q-values are stored.
    "replay_buffer_counts_period": 100, # Period at which replay buffer counts are stored.
    "rollouts_period": 100, # Period at which evaluation rollouts are performed.
    "num_rollouts": 50, # Number of rollouts to perform.
    "alpha_schedule_episodes": 5_000, # Learning rate anneal time.
    "alpha_init": 0.01, # Initial learning rate.
    "alpha_final": 0.001, # Final learning rate
    "gamma": 0.9, # Discount factor.
    "expl_eps_episodes": 5_000, # Exploration coefficient anneal time.
    "expl_eps_init": 1.0, # Initial exploration coefficient.
    "expl_eps_final": 0.0, # Final exploration coefficient.
    "replay_size": 20_000, # Replay buffer size.
    "batch_size": 32, # Batch size for model updates.
    "uniform_replay": True, # Whether to use a synthetically generated (uniform) replay buffer for training.
}


ORACLE_Q_VALS_DATA = {
    'mdp1': "four_state_mdp_val_iter_2022-11-24-10-33-28", # gamma = 0.9
}


def main(args):

    # Setup experiment data folder.
    exp_name = create_exp_name({'env_name': args['env_name'],
                                'algo': args['algo']})
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('run.py arguments:')
    print(args)

    # Store args copy. 
    f = open(exp_path + "/args.json", "w")
    json.dump(args, f)
    f.close()

    # Create four-state MDP.
    env = FourStateMDP()

    algo = LinearApproximator(env=env, alpha_init=args["alpha_init"], alpha_final=args["alpha_final"],
                                alpha_schedule_episodes=args["alpha_schedule_episodes"],
                                gamma=args["gamma"], expl_eps_init=args["expl_eps_init"],
                                expl_eps_final=args["expl_eps_final"], expl_eps_episodes=args["expl_eps_episodes"],
                                uniform_replay=args["uniform_replay"], replay_size=args["replay_size"],
                                batch_size=args["batch_size"])

    data = algo.train(num_episodes=args["num_episodes"], q_vals_period=args["q_vals_period"],
                replay_buffer_counts_period=args["replay_buffer_counts_period"],
                num_rollouts=args["num_rollouts"], rollouts_period=args["rollouts_period"])

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)


if __name__ == "__main__":
    main(args=ARGS)
