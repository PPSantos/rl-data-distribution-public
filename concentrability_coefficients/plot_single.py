import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import entropy

sns.set_style("darkgrid")
matplotlib.rcParams['text.usetex'] =  True
plt.rc('text.latex', preamble=r'\usepackage{pifont}')
matplotlib.rcParams.update({'font.size': 16})

ENV_NAME = "multiPathEnv"
EXP_ID = 'multiPathEnv_gamma=0.99_2023-11-24_14:10:22'
VMIN = 0.0
VMAX = 0.1

if __name__ == "__main__":

    print('EXP ID:', EXP_ID)

    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    # Open data.
    exp_data = np.load(f"data/{EXP_ID}.npy", allow_pickle=True)[()]
    print("Num. exps:", len(exp_data))

    # Find the exp. that achieved the lowest c value.
    c_vals = np.array([e["lowest_c_val"] for e in exp_data])
    print('c_vals:', c_vals )
    lowest_cval_idx = np.argmin(c_vals)
    exp_data = exp_data[lowest_cval_idx]

    print('Args:', exp_data["args"])
    print("Opt. mu entropy:", entropy(exp_data["opt_mu"]))
    print("Min opt. mu entry:", np.min(exp_data["opt_mu"]))
    print("Max opt. mu entry:", np.max(exp_data["opt_mu"]))

    # Plot c value across iterations.
    fig = plt.figure()
    fig.set_size_inches(5.0, 4.0)
    fig.tight_layout()
    xs = np.arange(len(exp_data["c_vals"])) * exp_data['args']['log_interval']
    plt.plot(xs, exp_data["c_vals"])
    plt.xlabel('Iteration')
    plt.ylabel(r'$C_3(\mu;\rho)$')
    plt.savefig('plots/c_vals_iter.pdf', bbox_inches='tight', pad_inches=0)

    if ENV_NAME == "gridEnv":
    
        fig, ax = plt.subplots(figsize=(4,4))
        
        norm = matplotlib.colors.Normalize(vmin=VMIN, vmax=VMAX)
        mesh = plt.pcolormesh(exp_data["opt_mu"].reshape(8,8), norm=norm)
        
        fig.colorbar(mesh,norm=norm)

        plt.grid(color='black')
        plt.savefig('plots/grid_plot.pdf', bbox_inches='tight', pad_inches=0)
