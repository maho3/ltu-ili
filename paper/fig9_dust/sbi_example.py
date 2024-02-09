import corner
import numpy as np
import h5py
import torch

import ili

num_dim = 4

gamma_min_arr = np.array([0.01, 0.02, 0.04, 0.04, 0.05, 0.06])
gamma_max_arr = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
tau_V_ISM_arr = np.array([0.1, 0.2, 0.3, 0.4])
tau_V_BC_arr = np.array([0.5, 0.6, 0.7, 0.8])

N_sims = (len(gamma_min_arr) * len(gamma_max_arr) *
          len(tau_V_ISM_arr) * len(tau_V_BC_arr))

theta = [None] * N_sims

for i, gamma_min in enumerate(gamma_min_arr):
    for j, gamma_max in enumerate(gamma_max_arr):
        for k, tau_V_ISM in enumerate(tau_V_ISM_arr):
            for l, tau_V_BC in enumerate(tau_V_BC_arr):
                index = i*len(gamma_max_arr)*len(tau_V_BC_arr)*len(tau_V_ISM_arr) +\
                    j*len(tau_V_BC_arr)*len(tau_V_ISM_arr) +\
                    k*len(tau_V_BC_arr) + l

                theta[index] = [gamma_min, gamma_max, tau_V_ISM, tau_V_BC]


theta = torch.tensor(theta, dtype=torch.float32)

with h5py.File('simba_mags.hdf5', 'r') as hf:
    mags = hf['mags'][:]

# rearrange so dust model first, then gals, then band
mags = np.moveaxis(mags, [1, 0], [0, 1])

# calculate LF


def calc_df(_x, volume, massBinLimits):
    hist, dummy = np.histogram(_x, bins=massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])

    phi_sigma = (np.sqrt(hist) / volume) /\
                (massBinLimits[1] - massBinLimits[0])  # Poisson errors

    return phi, phi_sigma, hist


binLimits = np.linspace(-31, -26, 10)
bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1])/2

phi = [None] * N_sims
h = 0.68

for i, _m in enumerate(mags):
    phi[i] = np.log10(calc_df(_m[:, 2], (100/h)**3, binLimits)[0])
    phi[i][np.isinf(phi[i])] = -10

phi = torch.tensor(phi, dtype=torch.float32)

loader = ili.dataloaders.NumpyLoader(x=phi, theta=theta)

prior = ili.utils.Uniform(low=[gamma_min_arr.min(),
                               gamma_max_arr.min(),
                               tau_V_ISM_arr.min(),
                               tau_V_BC_arr.min()],
                          high=[gamma_min_arr.max(),
                                gamma_max_arr.max(),
                                tau_V_ISM_arr.max(),
                                tau_V_BC_arr.max()]
                          )

nets = [
    ili.utils.load_nde_sbi(engine='NPE', model='maf')
]

runner = ili.inference.InferenceRunner.load(
    backend='sbi',
    engine='NPE',
    nets=nets,
    prior=prior,
)

posterior, _ = runner(loader)

idx = 100
x_o = np.log10(calc_df(mags[idx, :, 2], (25./h)**3, binLimits)[0])
x_o[np.isinf(x_o)] = -10

posterior_samples = posterior.sample((10000,), x=x_o)

# plot posterior samples
# _ = analysis.pairplot(
#     posterior_samples,  # limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(5, 5)
# )


# torch.tensor(Y[idx])

N_samps = int(4e4)

posterior_samples = posterior.sample(
    (N_samps,), x=torch.Tensor(x_o).to('cuda'))
posterior_samples = posterior_samples.detach().cpu().numpy()

fig = corner.corner(
    posterior_samples,
    labels=cam.cond_params[1:],
    quantiles=[0.16, 0.5, 0.84],  # 0.5
    show_titles=True,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False
)

corner.overplot_lines(fig, Y[idx], color="C1")
corner.overplot_points(fig, Y[idx][None], marker="s", color="C1")

ax = fig.add_axes([0.6, 0.6, 0.3, 0.3])

bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1])/2
with np.errstate(divide='ignore'):
    mean = torch.tensor(
        np.median(np.array(posterior_samples), axis=0), device='cuda')
    ax.plot(bins, np.mean([np.log10(generate_hmf(mean)) for i in np.arange(10)], axis=0),
            label='Posterior median', zorder=2)
    ax.plot(bins, np.log10(x_o), label='True', zorder=3)

    [ax.plot(bins, np.log10(generate_hmf(torch.tensor(posterior_samples[i], device='cuda'))),
             alpha=0.1, color='black', zorder=0) for i in np.arange(100)]

ax.set_ylim(-4.5, -0.5)
ax.set_xlim(10, 14.4)
ax.grid(alpha=0.1)
ax.set_xlabel('$M_{\mathrm{halo}} \,/\, \mathrm{M_{\odot}}$')
ax.set_ylabel('$\phi \,/\, \mathrm{Mpc^{-3} \; dex^{-1}}$')
ax.legend()

plt.savefig(f'plots/sbi_halo_example_{idx}.png', dpi=250)
plt.close()
