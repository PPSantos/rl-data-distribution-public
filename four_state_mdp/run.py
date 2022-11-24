"""
    Four-state MDP experiments.
"""
import os
import json
import time
import pathlib
import multiprocessing as mp
import multiprocessing.context as ctx
ctx._force_start_method('spawn')

from algos.linear_approximator import LinearApproximator
from four_state_mdp.env import FourStateMDP

from utils.strings import create_exp_name
from utils.json_utils import NumpyEncoder

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/'

###########################################################################
ARGS = {
    "env_name": "four_state_mdp",
    "algo": "linear_approximator",

    "num_runs": 6, # Number of runs to perform.
    "num_processors": 3, # Number of processors to use.
    "num_episodes": 20_000, # Number of episodes to train.
    "q_vals_period": 20, # Period at which Q-values are stored.
    "replay_buffer_counts_period": 20, # Period at which replay buffer counts are stored.
    "rollouts_period": 20, # Period at which evaluation rollouts are performed.
    "num_rollouts": 5, # Number of rollouts to perform.
    
    "alpha": 1e-04, # Learning rate.
    "gamma": 1.0, # Discount factor.
    "expl_eps_episodes": 20_000, # Exploration coefficient anneal time.
    "expl_eps_init": 0.05, # Initial exploration coefficient.
    "expl_eps_final": 0.05, # Final exploration coefficient.
    "replay_size": 2_000_000, # Replay buffer size.
    "batch_size": 100, # Batch size for model updates.
    "uniform_replay": True, # Whether to use a synthetically generated (uniform) replay buffer for training.
}


ORACLE_Q_VALS_DATA = {
    'mdp1': "four_state_mdp_val_iter_2022-11-24-11-31-49", # gamma = 1.0
}


def train(train_args):

    time_delay, args = train_args

    time.sleep(time_delay)

    # Create four-state MDP.
    env = FourStateMDP()

    algo = LinearApproximator(env=env, alpha=args["alpha"], gamma=args["gamma"], expl_eps_init=args["expl_eps_init"],
                                expl_eps_final=args["expl_eps_final"], expl_eps_episodes=args["expl_eps_episodes"],
                                uniform_replay=args["uniform_replay"], replay_size=args["replay_size"],
                                batch_size=args["batch_size"])

    data = algo.train(num_episodes=args["num_episodes"], q_vals_period=args["q_vals_period"],
                replay_buffer_counts_period=args["replay_buffer_counts_period"],
                num_rollouts=args["num_rollouts"], rollouts_period=args["rollouts_period"])
    
    return data

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

    # Adjust the number of processors if necessary.
    if args['num_processors'] > mp.cpu_count():
        args['num_processors'] = mp.cpu_count()
        print(f"Downgraded the number of processors to {args['num_processors']}.")

    # Train agent(s).
    with mp.Pool(processes=args['num_processors']) as pool:
        train_data = pool.map(train, [(2*t, args) for t in range(args['num_runs'])])
        pool.close()
        pool.join()

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(train_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('Experiment ID:', exp_name)


if __name__ == "__main__":
    main(args=ARGS)
