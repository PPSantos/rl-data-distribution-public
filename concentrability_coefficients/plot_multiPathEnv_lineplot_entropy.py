import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import entropy

sns.set_style("darkgrid")
matplotlib.rcParams['text.usetex'] =  True
plt.rc('text.latex', preamble=r'\usepackage{pifont}')
matplotlib.rcParams.update({'font.size': 16})

plt.rcParams["axes.axisbelow"] = False
import matplotlib.colors as colors


""" EXP_IDS = {
    0.2: 'multiPathEnv_gamma=0.2_2023-11-23_14:29:31',
    0.4: 'multiPathEnv_gamma=0.4_2023-11-23_14:36:01',
    0.6: 'multiPathEnv_gamma=0.6_2023-11-23_14:42:31',
    0.8: 'multiPathEnv_gamma=0.8_2023-11-23_14:48:57',
} """

# New experiments with M tuned according to gamma.
EXP_IDS = {
    0.2: 'multiPathEnv_gamma=0.2_2023-11-23_17:27:46',
    0.4: 'multiPathEnv_gamma=0.4_2023-11-23_17:33:33',
    0.6: 'multiPathEnv_gamma=0.6_2023-11-23_17:43:01',
    0.8: 'multiPathEnv_gamma=0.8_2023-11-23_18:02:17',
    #0.95: 'multiPathEnv_gamma=0.95_2023-11-24_02:29:59',
    0.99: 'multiPathEnv_gamma=0.99_2023-11-24_14:10:22',
}


if __name__ == "__main__":

    entropies_data = {}

    for gamma, data_f in EXP_IDS.items():
        print("gamma=", gamma)

        # Open data and ind the exp. that achieved the lowest c value.
        exp_data = np.load("data/" + EXP_IDS[gamma] + ".npy", allow_pickle=True)[()]
        c_vals = np.array([e["lowest_c_val"] for e in exp_data])
        #print("c_vals adam:", c_vals)
        exp_data = exp_data[np.argmin(c_vals)]

        entropies_data[gamma] = entropy(exp_data["opt_mu"])

        print("opt_mu:", exp_data["opt_mu"])
        print("min opt_mu:", np.min(exp_data["opt_mu"]))
        print("max opt_mu:", np.max(exp_data["opt_mu"]))

    print("entropies_data:", entropies_data)

    unif_mu = np.ones(5*5+2) / (5*5+2)
    print("Unif mu entropy:", entropy(unif_mu))

    fig = plt.figure()
    fig.set_size_inches(5.0, 4.0)
    fig.tight_layout()

    xs = []
    ys = []
    for (gamma, data_entropy) in entropies_data.items():
        xs.append(gamma)
        ys.append(data_entropy / entropy(unif_mu))

    plt.plot(xs, ys, zorder=10)
        
    plt.ylabel(r'$\mathcal{H}(\hat{\mu}) / \mathcal{H}(\mathcal{U})$')
    #plt.legend(loc=4)
    plt.xlabel(r'$\gamma$')
    plt.xlim([0.1,1.05])
    plt.xticks([0.2,0.4,0.6,0.8,0.99],[r"$0.2$",r"$0.4$",r"$0.6$",r"$0.8$",r"$0.99$"])
    plt.ylim([0.7,1.05])

    plt.savefig('plots/multiPathEnv_opt_dist_lineplot.pdf', bbox_inches='tight', pad_inches=0)
