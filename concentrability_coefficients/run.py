import os
import time
import datetime
import multiprocessing as mp
import multiprocessing.context as ctx
ctx._force_start_method('spawn')

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


from envs import GridEnv, multi_path_mdp_get_P
from opt import Adam

def log_sum_exp(x):
    return np.log(np.sum(np.exp(x)))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def eval_mu(mdp, mu, rho, M, use_soft_max):

    # Evaluates concentrability coefficient using backward induction.
    
    nS = mdp["state_space_size"]
    nA = mdp["action_space_size"]
    P = mdp["P"]
    gamma = mdp["gamma"]

    opt_policy = np.zeros((M, nS, nA))
    U = np.zeros((M+1, nS))

    U[-1,:] = np.sqrt((1.0/mu)) # (M, nS) Last step reward.

    for t in range(M-1, -1, -1): # M-1, ..., 0 (included).
        
        Q_t = np.zeros((nA, nS))
        for action in range(nA):
            Q_t[action,:] = np.dot(P[action], U[t+1,:])
        U[t,:] = np.max(Q_t, axis=0)

        policy_t = np.zeros((nS, nA))
        policy_act_idxs = np.argmax(Q_t, axis=0)
        policy_t[np.arange(nS), policy_act_idxs] = 1.0
        opt_policy[t,:,:] = policy_t

    sum_val = 0.0
    for m in range(1,M+1):
        sum_val += m*gamma**(m-1)*np.dot(rho, U[M-m])
        
    if use_soft_max:
        max_val = log_sum_exp(1.0/(mu**0.5))
    else:
        # hard max.
        max_val = np.max(1.0/(mu**0.5))
        
    c_val = (1-gamma)**2 * sum_val + gamma**M * (M-gamma*M+1)*max_val 

    return c_val, U, opt_policy

def get_mu_grad(mdp, opt_policies, mu, rho, M, use_soft_max):
    
    # Computes gradient w.r.t. `mu` variable.
    gamma = mdp["gamma"]
    nS = mdp["state_space_size"]
    nA = mdp["action_space_size"]
    P = mdp["P"]

    def build_P_pi_m(policy):
        P_pi_m = np.zeros((nS,nS))
        for a in range(nA):
            P_pi_m += np.dot(np.diag(policy[:,a]), P[a])
        return P_pi_m

    # Build P_pi_1.
    P_pi_1 = build_P_pi_m(policy=opt_policies[-1])

    # Variable `beta` holds the expected state
    # distribution at a given timestep.
    beta = np.dot(rho, P_pi_1)

    mu_grad = np.zeros(nS)

    norm = np.sqrt(np.dot(beta**2, 1.0 / mu))
    mu_grad += (-1) * (beta**2 / (2 * mu**2 * norm))
    
    for m in range(2, M+1):
        
        # Build P_pi_m.
        P_pi_m = build_P_pi_m(policy=opt_policies[-m])
        beta = np.dot(beta, P_pi_m)

        norm = np.sqrt(np.dot(beta**2, 1.0 / mu))
        mu_grad += (m * gamma**(m-1)) * (-1) * (beta**2 / (2 * mu**2 * norm))

    if use_soft_max:
        max_grad = softmax(1.0/(mu**0.5)) * (-1.0/(2*mu**1.5))
    else:
        max_idx = np.argmax(1.0/mu)
        mask = np.zeros(nS)
        mask[max_idx] = 1.0
        max_grad = mask *((-1.0)/(2*mu**1.5))

    mu_grad = (1-gamma)**2 * mu_grad + gamma**M * (M-gamma*M + 1) * max_grad
    
    return mu_grad

def optimize_mu(mdp, mu_init, rho, M, K=10_000,
                alpha=1e-04, alpha_decay=0.0, 
                optimizer='sgd', use_soft_max=False,
                log_interval=500):
    
    # Arguments:
    # - mu_init: initial mu distribution.
    # - rho: norm distribution.
    # - M: horizon.
    # - K: number of iterations.
    # - alpha: learning rate.
    # - alpha_decay: learning rate decay parameter.
    # - optimizer: 'sgd' or 'adam'.
    # - use_soft_max: whether to use soft max instead
    #       of the hard max (second term in the objective).
    # - log_interval: timesteps between logs.
    
    print('Optimizing for optimal mu...')
    print("Arguments:")
    print('mu_init:', mu_init)
    print('rho:', rho)
    print('K:', K)
    print('M:', M)
    print('alpha:', alpha)
    print('Log interval', log_interval)

    c_val, _, _ = eval_mu(mdp, mu_init, rho=rho, M=M, use_soft_max=use_soft_max)
    print('Initial C value:', c_val)
    
    if optimizer not in ['sgd', 'adam']:
        raise ValueError('Unknown optimizer.')
    
    if optimizer == 'adam':
        adam_opt = Adam(lr=alpha, decay=alpha_decay)
    
    mu = mu_init

    c_vals = [c_val]
    mus = [mu]
    opt_mu = mu
    lowest_c_val = c_val
    
    mu_grad_means = []
    mu_grad_stds = []

    for k in tqdm(range(K)):
        
        if k % log_interval == 0:
            print(f'Iteration {k}: C-value={c_val}')

        c_val, _, opt_policies = eval_mu(mdp, mu, rho=rho, M=M, use_soft_max=use_soft_max)

        if c_val < lowest_c_val:
            lowest_c_val = c_val
            opt_mu = mu

        mu_grad = get_mu_grad(mdp, opt_policies=opt_policies, mu=mu, rho=rho, M=M, use_soft_max=use_soft_max)
        
        if k % log_interval == 0:
            mu_grad_means.append(np.mean(mu_grad))
            mu_grad_stds.append(np.std(mu_grad))

        if optimizer == 'sgd':
            mu = mu - alpha * mu_grad
        elif optimizer == 'adam':
            mu = adam_opt.get_update([mu], [mu_grad])[0]
        
        mu = projection_simplex_sort(mu, z=1)

        # Ensure mu > 0 in order to prevent numerical instabilities.
        mu += 1e-07
        mu /= np.sum(mu)

        if k % log_interval == 0:
            c_vals.append(c_val)
            mus.append(mu)

    c_val, _, _ = eval_mu(mdp, mu, rho=rho, M=M, use_soft_max=use_soft_max)
    print('Final C value:', c_val)
    print('Final mu entropy :', entropy(mu))
    
    return opt_mu, lowest_c_val, mus, c_vals, mu_grad_means, mu_grad_stds

def run(run_args):

    time_delay, args = run_args

    time.sleep(time_delay)
    np.random.seed(time_delay)

    # Create env.
    if args["env"] == "gridEnv":

        rho = np.zeros((8,8))
        rho[0,0] = 1.0
        rho = rho.flatten()
        grid_env = GridEnv(width=8,height=8,starting_state_dist=rho, rewards=[(2,2)],
                           time_limit=20, stochasticity=args["env_stochasticity"] )
        mdp = {"state_space_size": grid_env.get_state_space_size(),
                "action_space_size": grid_env.get_action_space_size(),
                "P": grid_env.get_P(),
                "R": grid_env.get_R(),
                "gamma": args["gamma"]}
        
    elif args["env"] == "multiPathEnv":
        rho = np.zeros((5*5+2))
        rho[0] = 1.0
        mdp = {"state_space_size": 5*5+2,
            "action_space_size": 5,
            "P": multi_path_mdp_get_P(),
            "R": None, # (irrelevant from the PoV of concentrability).
            "gamma": args["gamma"]}
    else:
        raise ValueError("Unknown environment.")
    
    # Sample initial mu distribution.
    mu_init = np.random.rand(mdp["state_space_size"])
    mu_init = mu_init / np.sum(mu_init)
    print('*'*30)
    print(mu_init)

    # Otimize mu.
    opt_mu, lowest_c_val, mus, c_vals, mu_grad_means, mu_grad_stds = \
                                            optimize_mu(mdp,
                                                    mu_init=mu_init,
                                                    rho=rho,
                                                    M=args["M"],
                                                    K=args["K"],
                                                    alpha=args["alpha"],
                                                    alpha_decay=args["alpha_decay"],
                                                    optimizer=args["optimizer"],
                                                    use_soft_max=args["use_softmax"],
                                                    log_interval=args["log_interval"])
    
    run_data = {}
    run_data["opt_mu"] = opt_mu
    run_data["lowest_c_val"] = lowest_c_val
    run_data["mus"] = mus
    run_data["c_vals"] = c_vals
    run_data["mu_grad_means"] = mu_grad_means
    run_data["mu_grad_stds"] = mu_grad_stds
    run_data["args"] = args
    
    return run_data


def main(args):

    # Adjust the number of processors if necessary.
    if args["num_processors"] > mp.cpu_count():
        args["num_processors"] = mp.cpu_count()
        print(f"Downgraded the number of processors to {args['num_processors']}.")

    # Train agent(s).
    with mp.Pool(processes=args["num_processors"]) as pool:
        data = pool.map(run, [(2*t, args) for t in range(args['num_runs'])])
        pool.close()
        pool.join()

    # Dump data.
    if not os.path.exists("data/"):
        os.makedirs("data/")
    str_id = str(datetime.date.today()) + '_' + time.strftime("%H:%M:%S", time.localtime())
    exp_name = args["env"] + "_gamma=" + str(args["gamma"]) + "_"
    if args["env"] == "gridEnv":
        exp_name += "sto=" + str(args["env_stochasticity"]) + "_"
    exp_name += str_id
    print("exp_name:", exp_name)

    np.save(f'data/{exp_name}.npy', data)

    return exp_name

if __name__ == "__main__":

    # Script parameters.
    args = {}
    args["num_runs"] = 2
    args["num_processors"] = 2
    args["log_interval"] = 500
    args["gamma"] = 0.2
    args["env"] = "multiPathEnv" # gridEnv or multiPathEnv.
    args["env_stochasticity"] = 0.0 # gridEnv parameter only.

    args["M"] = 10
    args["K"] = 1_000
    args["alpha"] = 1e-07
    args["alpha_decay"] = 0.0
    args["optimizer"] = "adam"
    args["use_softmax"] = False

    main(args)
