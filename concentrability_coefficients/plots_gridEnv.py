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

ENV_STOCHASTICITY_TO_PLOT = 0.4
VMIN = 0.0
VMAX = 0.1
VMIN_NORMALIZED = 0.5
VMAX_NORMALIZED = 5.0

EXP_IDS_ADAM = {
    
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

}

EXP_IDS_SGD = {
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

}



if __name__ == "__main__":

    exps_data = {}
    for gamma in EXP_IDS_ADAM[ENV_STOCHASTICITY_TO_PLOT].keys():
        print('gamma=', gamma)

        # Open data and ind the exp. that achieved the lowest c value.
        exp_data = np.load("data/" + EXP_IDS_ADAM[ENV_STOCHASTICITY_TO_PLOT][gamma] + ".npy", allow_pickle=True)[()]
        c_vals = np.array([e["lowest_c_val"] for e in exp_data])
        print("c_vals adam:", c_vals)
        lowest_cval_adam = np.min(c_vals)
        exp_data_adam = exp_data[np.argmin(c_vals)]

        exp_data = np.load("data/" + EXP_IDS_SGD[ENV_STOCHASTICITY_TO_PLOT][gamma] + ".npy", allow_pickle=True)[()]
        c_vals = np.array([e["lowest_c_val"] for e in exp_data])
        print("c_vals sgd:", c_vals)
        lowest_cval_sgd = np.min(c_vals)
        exp_data_sgd = exp_data[np.argmin(c_vals)]

        if lowest_cval_sgd < lowest_cval_adam:
            print('Lowest c_val is from SGD.')
            exps_data[gamma] = exp_data_sgd
        else:
            print('Lowest c_val is from ADAM.')
            exps_data[gamma] = exp_data_adam        

    print('\n')
    for gamma, data_dict in exps_data.items():
        print('gamma=', gamma)
        unif_mu = np.ones((8,8)) / (8*8)
        print('max mu=', np.max(data_dict["opt_mu"] ))
        print('min mu=', np.min(data_dict["opt_mu"] ))
        print('max normalized mu=', np.max(data_dict["opt_mu"] / unif_mu.flatten() ))
        print('min normalized mu=', np.min(data_dict["opt_mu"] / unif_mu.flatten() ))

    print(len(exps_data))

    # Normalized grid plot.
    fig, axes = plt.subplots(1, len(exps_data), figsize=(len(exps_data)*3,3))
    fig.tight_layout()

    colors_undersea = plt.cm.cividis(np.linspace(0, 0.5, 256))
    colors_land = plt.cm.cividis(np.linspace(0.5, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    terrain_map = colors.LinearSegmentedColormap.from_list(
        'terrain_map', all_colors)

    divnorm = colors.TwoSlopeNorm(vmin=VMIN_NORMALIZED, vcenter=1.0, vmax=VMAX_NORMALIZED)

    for ax, (gamma, data_dict) in zip(axes, exps_data.items()):
        
        normalized_mu = data_dict["opt_mu"].reshape(8,8) / unif_mu
        mesh = ax.pcolormesh(normalized_mu, norm=divnorm, cmap=terrain_map)

        ax.set_xticks(np.arange(1,8))
        ax.set_yticks(np.arange(1,8))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(r"$\gamma =$ " + str(gamma))
        ax.grid(True, color="white")

    bbox_ax = axes[0].get_position()
    cbar_ax = fig.add_axes([1.01, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
    cbar = fig.colorbar(mesh, cax=cbar_ax, norm=divnorm, ticks=[VMIN_NORMALIZED, 1.0, VMAX_NORMALIZED])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(r'$\hat{\mu}(s) / \mathcal{U}(s) $', rotation=270)

    plt.savefig(f'plots/opt_dist_normalized_sto={ENV_STOCHASTICITY_TO_PLOT}.pdf', bbox_inches='tight', pad_inches=0)

    # Unormalized plot.
    unif_mu = np.ones((8,8)) / (8*8)

    fig, axes = plt.subplots(1, len(exps_data), figsize=(len(exps_data)*3,3))
    fig.tight_layout()

    norm = matplotlib.colors.Normalize(vmin=VMIN, vmax=VMAX)

    for ax, (gamma, data_dict) in zip(axes, exps_data.items()):
        
        mu_dist = data_dict["opt_mu"].reshape(8,8)
        mesh = ax.pcolormesh(mu_dist, norm=norm, cmap="coolwarm")

        ax.set_xticks(np.arange(1,8))
        ax.set_yticks(np.arange(1,8))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(r"$\gamma =$ " + str(gamma))
        ax.grid(True, color="white")

    bbox_ax = axes[0].get_position()
    cbar_ax = fig.add_axes([1.01, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
    cbar = fig.colorbar(mesh, cax=cbar_ax, norm=norm, ticks=[0.0, 0.05])
    cbar.ax.set_ylabel(r'$\hat{\mu}(s)$', rotation=270)

    #plt.show()
    plt.savefig(f'plots/opt_dist_sto={ENV_STOCHASTICITY_TO_PLOT}.pdf', bbox_inches='tight', pad_inches=0)