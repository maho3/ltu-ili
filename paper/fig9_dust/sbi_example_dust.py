import numpy as np
import h5py
import pickle
import torch
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt

import corner
# from schwimmbad import MultiPool

from sbi.inference import prepare_for_sbi, simulate_for_sbi
import ili

from synthesizer.sed import Sed
from synthesizer.filters import FilterCollection
from synthesizer.conversions import lnu_to_absolute_mag
from synthesizer.load_data.load_simba import load_Simba

from sbi_dust_methods import get_colour_and_lf


directory = '/cosma7/data/dp004/dc-love2/codes/synthesizer-pipeline/Simba'

with h5py.File(f'{directory}/simba.hdf5', 'r') as hf:
    young_specs = hf['spectra/young_stellar'][:]
    old_specs = hf['spectra/old_stellar'][:]
    lam = hf['spectra/wavelength'][:]

young_specs = Sed(lam=lam, lnu=young_specs)
old_specs = Sed(lam=lam, lnu=old_specs)

# define a filter collection object
try:
    fc = FilterCollection(path='custom_filter_collection.hdf5')
except:
    fs = [f"SLOAN/SDSS.{f}" for f in ['u', 'g', 'r', 'i', 'z']]

    fc = FilterCollection(
        filter_codes=fs,
        new_lam=young_specs.lam
    )

    fc.write_filters('custom_filter_collection.hdf5')


binLimsLF = np.linspace(-25, -17, 10)  # r-band
binLimsColour = np.linspace(0, 1.3, 25)  # g-r
# binLimits = np.linspace(0, 3.5, 40) # u-z

prior = ili.utils.Uniform(
    low=torch.tensor([0.01, 0.4, 0.3]),
    high=torch.tensor([0.6, 2.0, 2.0]),
)

get_partial = partial(
    get_colour_and_lf,
    young_specs=young_specs,
    old_specs=old_specs,
    fc=fc,
    binLimsColour=binLimsColour,
    binLimsLF=binLimsLF
)
simulator, prior = prepare_for_sbi(get_partial, prior)

theta, x = simulate_for_sbi(
    simulator,
    proposal=prior,
    num_simulations=1000,
    num_workers=28,
)
loader = ili.dataloaders.NumpyLoader(x=x, theta=theta)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

ax1.plot(binLimsColour[:-1],
         x[:, :len(binLimsColour)-1].T, color='k', alpha=0.1)
ax2.plot(binLimsLF[:-1], x[:, len(binLimsColour)-1:].T, color='k', alpha=0.1)

plt.savefig('test.png')
plt.close()

# np.savetxt('data/simulations.txt', np.hstack([theta, x]))
# dat = torch.tensor(np.loadtxt('data/simulations_colour_lf.txt'), dtype=torch.float32)
# theta, x = dat[:, :3], dat[:, 3:]

just_colours = x[:, :len(binLimsColour)-1]
just_lf = x[:, len(binLimsColour)-1:]
# colours_and_lf = x

nets = [
    ili.utils.load_nde_sbi(engine='NPE', model='maf')
]

for _x, label in zip([just_colours, just_lf], ['colour', 'lf']):

    runner = ili.inference.InferenceRunner.load(
        backend='sbi',
        engine='NPE',
        nets=nets,
        prior=prior,
    )

    posterior, _ = runner(loader)

    with open(f"data/{label}_posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)


theta_o = np.random.uniform(low=[0.01, 0.3, 0.5], high=[0.5, 1.5, 2.0])
# x_o = apply_dust(theta_o, specs, fc, binLimits)
# , young_specs, old_specs, fc, binLimsColour, binLimsLF)
x_o = get_partial(theta_o)


# x_o += np.random.normal(0, scale=100, size=len(x_o)).astype(int)
# x_o[x_o < 0] = 0.

"""
Plot multiple posteriors
"""
label = 'colour_lf'
with open(f"data/{label}_posterior.pkl", "rb") as handle:
    posterior = pickle.load(handle)

_N = 50
dat = np.zeros((_N, 3, 3))
theta_o = np.zeros((_N, 3))
for i in np.arange(_N):
    theta_o[i] = np.random.uniform(low=[0.01, 0.3, 0.5], high=[0.5, 1.5, 2.0])
    x_o = get_partial(theta_o[i])
    posterior_samples = posterior.sample((10000,), x=x_o)
    posterior_samples = posterior_samples.detach().cpu().numpy()

    for j in np.arange(3):
        dat[i, j, :] = np.percentile(
            posterior_samples[:, j], q=[15.9, 50, 84.1])

np.savetxt('data/multi_percentile_samples.txt', dat.reshape(_N, 9))
np.savetxt('data/theta_o_samples.txt', theta_o)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 9))
plt.subplots_adjust(hspace=0.3)

for i in np.arange(50):
    for j, ax in enumerate([ax1, ax2, ax3]):
        percs = dat[i, j]
        ax.errorbar(theta_o[i, j], percs[1], fmt='.', color='C0',
                    yerr=np.array([percs[1] - percs[0], percs[2] - percs[1]])[:, None])


for ax, low, high in zip([ax1, ax2, ax3], [0.01, 0.3, 0.5], [0.5, 1.5, 2.0]):
    ax.plot([low, high], [low, high], linestyle='dashed', color='black')
    ax.set_xlabel('True')
    ax.set_ylabel('Posterior')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

ax1.text(0.05, 0.9, '$\\tau_{\mathrm{ISM}}$', transform=ax1.transAxes)
ax2.text(0.05, 0.9, '$\\tau_{\mathrm{BC}}$', transform=ax2.transAxes)
ax3.text(0.05, 0.9, '$\\alpha$', transform=ax3.transAxes)

plt.savefig('sbi_multi.png', dpi=250)
plt.close()


"""
Corner plot of a single example
"""

theta_o = np.random.uniform(low=[0.01, 0.3, 0.5], high=[0.5, 1.5, 2.0])
x_o = get_partial(theta_o)

fig = plt.figure(figsize=(8, 8))

for i, (label, _x_o) in enumerate(zip(
    ['colour', 'lf', 'colour_lf'],
    [x_o[:len(binLimsColour)-1], x_o[len(binLimsColour)-1:], x_o],
)):

    with open(f"data/{label}_posterior.pkl", "rb") as handle:
        posterior = pickle.load(handle)

    posterior_samples = posterior.sample((10000,), x=_x_o)
    posterior_samples = posterior_samples.detach().cpu().numpy()

    corner.corner(
        posterior_samples,
        labels=['$\\rm \\tau_{v}^{ISM}$',
                '$\\rm \\tau_{v}^{BC}$', '$\\alpha$'],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        plot_datapoints=False,
        plot_density=False,
        fig=fig,
        color=f'C{i}',
    )

corner.overplot_lines(fig, theta_o, color="red")
corner.overplot_points(fig, theta_o[None], marker="s", color="red")

labels = ['$g - r$', '$M_r$', '$g - r$ & $M_r$']
plt.legend(
    handles=[
        mpl.lines.Line2D([], [], color=f'C{i}', label=labels[i])
        for i in range(3)
    ],
    fontsize=15, frameon=False,
    bbox_to_anchor=(1, 3), loc="upper right"
)

# plt.savefig(f'sbi_example.png', dpi=250); plt.close()

# ax_lf = fig.add_axes([0.5, 0.75, 0.2, 0.2])
# ax_colours = fig.add_axes([0.75, 0.5, 0.2, 0.2])
#
# mask = np.zeros(x.shape[1], dtype=bool)
# mask_colours, mask_lf = mask.copy(), mask.copy()
# mask_colours[:len(binLimsColour)-1] = True
# mask_lf[len(binLimsColour)-1:] = True
#
# for i, (label, ax, binLimits, mask) in enumerate(zip(
#     ['colour', 'lf'],
#     [ax_colours, ax_lf],
#     [binLimsColour, binLimsLF],
#     [mask_colours, mask_lf],
# )):
#
#     with open(f"data/{label}_posterior.pkl", "rb") as handle:
#         posterior = pickle.load(handle)
#
#     posterior_samples = posterior.sample((10000,), x=x_o[mask])
#     posterior_samples = posterior_samples.detach().cpu().numpy()
#
#     bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1])/2
#     ax.plot(bins, x_o[mask], label='True', zorder=3, color='C4')
#     mean = torch.tensor(np.median(np.array(posterior_samples), axis=0))
#     posterior_x = apply_dust_partial(mean)[mask]
#
#     print(theta_o, mean)
#     print(x_o[mask], posterior_x)
#
#     ax.plot(bins, posterior_x, label=f'Posterior median {label}',
#             linestyle='dashed', color=f'C{i}')
#
#     # with MultiPool() as pool:
#     #     plot_x = pool.map(apply_dust_partial, posterior_samples[:10])
#     #
#     # [ax.plot(bins, x[mask], alpha=0.1, color='black', zorder=0) for x in plot_x];
#     [ax.plot(bins, apply_dust_partial(_p)[mask], alpha=0.1, color='black', zorder=0)
#             for _p in posterior_samples[:10]];
#
#     # ax.set_ylim(-4.5, -0.5);
#     # ax.set_xlim(10, 14.4)
#     # ax.set_xlabel('$M_{\mathrm{halo}} \,/\, \mathrm{M_{\odot}}$')
#     ax.grid(alpha=0.1)

# ax_lf.set_ylabel('$\phi \,/\, \mathrm{Mpc^{-3} \; dex^{-1}}$')
# ax_colours.set_xlabel('$g - r$')
# ax_lf.legend()

# plt.show()
plt.savefig(f'example.png', dpi=250)
plt.close()
