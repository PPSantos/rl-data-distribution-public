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

ENV_STOCHASTICITY_TO_PLOT = 0.0
VMIN = 0.0
VMAX = 0.1
VMIN_NORMALIZED = 0.1
VMAX_NORMALIZED = 7.0

EXP_IDS = {
    
    # stochasticity = 0.0
    0.0: {
        0.2: 'gridEnv_gamma=0.2_sto=0.0_2023-11-22_14:53:17',
        0.4: 'gridEnv_gamma=0.4_sto=0.0_2023-11-22_15:34:21',
        0.6: 'gridEnv_gamma=0.6_sto=0.0_2023-11-22_16:11:32',
        0.8: 'gridEnv_gamma=0.8_sto=0.0_2023-11-22_16:38:52',
    },

    # stochasticity = 0.2
    0.2: {
        #0.2: ,
        #0.4: ,
        #0.6: ,
        #0.8: ,
    },

    # stochasticity = 0.4
    0.4: {
        #0.2: ,
        #0.4: ,
        #0.6: ,
        #0.8: ,
    },

}


if __name__ == "__main__":

    exps_data = {}
    for gamma, data_f in EXP_IDS[ENV_STOCHASTICITY_TO_PLOT].items():
        print('gamma=', gamma)

        # Open data.
        exp_data = np.load("data/" + data_f + ".npy", allow_pickle=True)[()]

        # Find the exp. that achieved the lowest c value.
        c_vals = np.array([e["lowest_c_val"] for e in exp_data])
        print("c_vals", c_vals)
        lowest_cval_idx = np.argmin(c_vals)
        exp_data = exp_data[lowest_cval_idx]

        exps_data[gamma] = exp_data

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

    divnorm = colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=5.0)

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
    cbar = fig.colorbar(mesh, cax=cbar_ax, norm=divnorm, ticks=[0.0, 1.0, 5.0])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(r'$\hat{\mu}(s) / \mathcal{U}(s) $', rotation=270)

    plt.savefig('plots/opt_dist_normalized.pdf', bbox_inches='tight', pad_inches=0)
