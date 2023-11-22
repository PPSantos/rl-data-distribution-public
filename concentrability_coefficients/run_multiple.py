from run import main

if __name__ == "__main__":

    # Script parameters.
    args = {}
    args["num_runs"] = 2
    args["num_processors"] = 2
    args["log_interval"] = 500
    # args["gamma"] = 0.2
    args["env"] = "gridEnv" # gridEnv or multiPathEnv.
    args["env_stochasticity"] = 0.0 # gridEnv parameter only.
    args["M"] = 30
    args["K"] = 1_000
    args["alpha"] = 1e-07
    args["alpha_decay"] = 0.0
    args["optimizer"] = "adam"
    args["use_softmax"] = False

    exp_names = []
    for gamma in [0.2,0.4,0.6,0.8]:
        print("gamma=", gamma)
        args["gamma"] = gamma
        exp_name = main(args)
        exp_names.append(exp_name)

    print(exp_names)

