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


PLOT_MODE = "avg" # "best" or "avg".

""" EXP_IDS_ADAM = {
    
    # stochasticity = 0.0
    0.0: {
        0.2: 'gridEnv_gamma=0.2_sto=0.0_2023-11-22_18:40:04',
        0.4: 'gridEnv_gamma=0.4_sto=0.0_2023-11-22_19:19:00',
        0.6: 'gridEnv_gamma=0.6_sto=0.0_2023-11-22_19:57:36',
        0.8: 'gridEnv_gamma=0.8_sto=0.0_2023-11-22_20:35:11',
    },

    # stochasticity = 0.2
    0.2: {
        0.2: 'gridEnv_gamma=0.2_sto=0.2_2023-11-22_23:03:40',
        0.4: 'gridEnv_gamma=0.4_sto=0.2_2023-11-22_23:58:56',
        0.6: 'gridEnv_gamma=0.6_sto=0.2_2023-11-23_00:55:46',
        0.8: 'gridEnv_gamma=0.8_sto=0.2_2023-11-23_01:52:55',
    },

    # stochasticity = 0.4
    0.4: {
        0.2: 'gridEnv_gamma=0.2_sto=0.4_2023-11-22_23:14:01',
        0.4: 'gridEnv_gamma=0.4_sto=0.4_2023-11-23_00:10:47',
        0.6: 'gridEnv_gamma=0.6_sto=0.4_2023-11-23_01:07:53',
        0.8: 'gridEnv_gamma=0.8_sto=0.4_2023-11-23_02:00:11',
    },

} """

# New experiments with M tuned according to gamma.
EXP_IDS_ADAM = {
    
    # stochasticity = 0.0
    0.0: {
        0.2: 'gridEnv_gamma=0.2_sto=0.0_2023-11-23_22:32:01',
        0.4: 'gridEnv_gamma=0.4_sto=0.0_2023-11-23_22:48:12',
        0.6: 'gridEnv_gamma=0.6_sto=0.0_2023-11-23_23:18:25',
        0.8: 'gridEnv_gamma=0.8_sto=0.0_2023-11-24_00:24:20',
    },

    # stochasticity = 0.2
    0.2: {
        0.2: 'gridEnv_gamma=0.2_sto=0.2_2023-11-23_22:33:47',
        0.4: 'gridEnv_gamma=0.4_sto=0.2_2023-11-23_22:50:29',
        0.6: 'gridEnv_gamma=0.6_sto=0.2_2023-11-23_23:20:19',
        0.8: 'gridEnv_gamma=0.8_sto=0.2_2023-11-24_00:26:08',
    },

    # stochasticity = 0.4
    0.4: {
        0.2: 'gridEnv_gamma=0.2_sto=0.4_2023-11-23_22:34:09',
        0.4: 'gridEnv_gamma=0.4_sto=0.4_2023-11-23_22:50:37',
        0.6: 'gridEnv_gamma=0.6_sto=0.4_2023-11-23_23:20:37',
        0.8: 'gridEnv_gamma=0.8_sto=0.4_2023-11-24_00:25:53',
    },

}

""" EXP_IDS_SGD = {
    # stochasticity = 0.0
    0.0: {
        0.2: 'gridEnv_gamma=0.2_sto=0.0_2023-11-23_10:09:00',
        0.4: 'gridEnv_gamma=0.4_sto=0.0_2023-11-23_11:00:39',
        0.6: 'gridEnv_gamma=0.6_sto=0.0_2023-11-23_11:52:36',
        0.8: 'gridEnv_gamma=0.8_sto=0.0_2023-11-23_12:44:59',
    },

    # stochasticity = 0.2
    0.2: {
        0.2: 'gridEnv_gamma=0.2_sto=0.2_2023-11-23_10:09:25',
        0.4: 'gridEnv_gamma=0.4_sto=0.2_2023-11-23_11:01:32',
        0.6: 'gridEnv_gamma=0.6_sto=0.2_2023-11-23_11:54:06',
        0.8: 'gridEnv_gamma=0.8_sto=0.2_2023-11-23_12:45:52',
    },

    # stochasticity = 0.4
    0.4: {
        0.2: 'gridEnv_gamma=0.2_sto=0.4_2023-11-23_10:10:00',
        0.4: 'gridEnv_gamma=0.4_sto=0.4_2023-11-23_11:02:16',
        0.6: 'gridEnv_gamma=0.6_sto=0.4_2023-11-23_11:53:42',
        0.8: 'gridEnv_gamma=0.8_sto=0.4_2023-11-23_12:45:32',
    },

} """


# New experiments with M tuned according to gamma.
EXP_IDS_SGD = {
    # stochasticity = 0.0
    0.0: {
        0.2: 'gridEnv_gamma=0.2_sto=0.0_2023-11-23_18:45:01',
        0.4: 'gridEnv_gamma=0.4_sto=0.0_2023-11-23_19:01:24',
        0.6: 'gridEnv_gamma=0.6_sto=0.0_2023-11-23_19:31:57',
        0.8: 'gridEnv_gamma=0.8_sto=0.0_2023-11-23_20:38:07',
    },

    # stochasticity = 0.2
    0.2: {
        0.2: 'gridEnv_gamma=0.2_sto=0.2_2023-11-23_18:45:50',
        0.4: 'gridEnv_gamma=0.4_sto=0.2_2023-11-23_19:02:12',
        0.6: 'gridEnv_gamma=0.6_sto=0.2_2023-11-23_19:32:08',
        0.8: 'gridEnv_gamma=0.8_sto=0.2_2023-11-23_20:38:11',
    },

    # stochasticity = 0.4
    0.4: {
        0.2: 'gridEnv_gamma=0.2_sto=0.4_2023-11-23_18:46:01',
        0.4: 'gridEnv_gamma=0.4_sto=0.4_2023-11-23_19:02:09',
        0.6: 'gridEnv_gamma=0.6_sto=0.4_2023-11-23_19:32:23',
        0.8: 'gridEnv_gamma=0.8_sto=0.4_2023-11-23_20:38:22',
    },

}


if __name__ == "__main__":

    entropies_data = {}
    for sto in EXP_IDS_ADAM.keys():
        print("sto=", sto)
        entropies_data[sto] = {}

        for gamma, data_f in EXP_IDS_ADAM[sto].items():
            print("gamma=", gamma)

            if PLOT_MODE == "best":
                # Open data and get the exp. that achieved the lowest c value.
                exp_data = np.load("data/" + EXP_IDS_ADAM[sto][gamma] + ".npy", allow_pickle=True)[()]
                c_vals = np.array([e["lowest_c_val"] for e in exp_data])
                print("c_vals adam:", c_vals)
                lowest_cval_adam = np.min(c_vals)
                exp_data_adam = exp_data[np.argmin(c_vals)]

                exp_data = np.load("data/" + EXP_IDS_SGD[sto][gamma] + ".npy", allow_pickle=True)[()]
                c_vals = np.array([e["lowest_c_val"] for e in exp_data])
                print("c_vals sgd:", c_vals)
                lowest_cval_sgd = np.min(c_vals)
                exp_data_sgd = exp_data[np.argmin(c_vals)]

                if lowest_cval_sgd < lowest_cval_adam:
                    print('Lowest c_val is from SGD.')
                    entropies_data[sto][gamma] = entropy(exp_data_sgd["opt_mu"])
                else:
                    print('Lowest c_val is from ADAM.')
                    entropies_data[sto][gamma] = entropy(exp_data_adam["opt_mu"])
            
            elif PLOT_MODE == "avg":
                exp_data = np.load("data/" + EXP_IDS_ADAM[sto][gamma] + ".npy", allow_pickle=True)[()]
                ents_adam = [entropy(e["opt_mu"]) for e in exp_data]
                print(ents_adam)

                exp_data = np.load("data/" + EXP_IDS_SGD[sto][gamma] + ".npy", allow_pickle=True)[()]
                ents_sgd = [entropy(e["opt_mu"]) for e in exp_data]
                print(ents_sgd)

                entropies_data[sto][gamma] = ents_sgd + ents_adam

                print(entropies_data[sto][gamma])

            else:
                raise ValueError("Unknown plotting mode.")

    print("entropies_data:", entropies_data)

    unif_mu = np.ones((8,8)) / (8*8)
    print("Unif mu entropy:", entropy(unif_mu.flatten()))

    fig = plt.figure()
    fig.set_size_inches(5.0, 4.0)
    fig.tight_layout()

    for sto in entropies_data.keys():
        xs = []
        ys = []
        for (key, data_entropy) in entropies_data[sto].items():
            xs.append(key)
            if PLOT_MODE == "best":
                ys.append(data_entropy / entropy(unif_mu.flatten()))
            elif PLOT_MODE == "avg":
                ys.append(np.mean(data_entropy) / entropy(unif_mu.flatten()))
            else:
                raise ValueError("Unknown plotting mode.")

        plt.plot(xs, ys, zorder=10, label=r"$\zeta =$ " + str(sto))
        
    plt.ylabel(r'$\mathcal{H}(\hat{\mu}) / \mathcal{H}(\mathcal{U})$')
    plt.legend(loc=4)
    plt.xlabel(r'$\gamma$')
    plt.xlim([0.1,0.9])
    plt.ylim([0.7,1.05])

    plt.savefig(f'plots/gridEnv_opt_dist_lineplot_{PLOT_MODE}.pdf', bbox_inches='tight', pad_inches=0)
