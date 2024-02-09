import numpy as np
import h5py
import torch
from functools import partial
import matplotlib.pyplot as plt

import corner

from sbi.inference import prepare_for_sbi, simulate_for_sbi
import ili

from synthesizer.sed import Sed
from synthesizer.filters import FilterCollection
from synthesizer.conversions import lnu_to_absolute_mag
from synthesizer.dust.attenuation import PowerLaw, Calzetti2000
from synthesizer.load_data.load_simba import load_Simba

from schwimmbad import MultiPool


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


# binLimits = np.linspace(-25, -17, 10) # r-band
binLimits = np.linspace(0, 1.2, 21)  # g-r

# def calc_df(_x, volume, massBinLimits):
#     hist, dummy = np.histogram(_x, bins=massBinLimits)
#     hist = np.float64(hist)
#     phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])
#
#     phi_sigma = (np.sqrt(hist) / volume) /\
#                 (massBinLimits[1] - massBinLimits[0]) # Poisson errors
#
#     return phi, phi_sigma, hist


"""
Simple screen model test
"""
# def apply_dust(theta, specs, fc, binLimits):


def apply_dust(theta, young_specs, old_specs, fc, binLimits):
    # tau_v, slope = theta
    tau_v_ism, tau_v_bc, slope = theta
    # tau_v, slope, cent_lam, ampl, gamma = theta
    # _spec = specs.apply_attenuation(
    #     tau_v=float(tau_v),
    #     dust_curve=PowerLaw(slope=np.array(slope)),
    #     # dust_curve=Calzetti2000(
    #     #     slope=np.array(slope),
    #     #     cent_lam=cent_lam,
    #     #     ampl=ampl,
    #     #     gamma=gamma,
    #     # )
    # )

    y_specs = young_specs.apply_attenuation(
        tau_v=float(tau_v_bc),
        dust_curve=PowerLaw(slope=np.array(slope)),
    )
    o_specs = old_specs.apply_attenuation(
        tau_v=float(tau_v_ism),
        dust_curve=PowerLaw(slope=np.array(slope)),
    )

    _spec = y_specs + o_specs

    # _spec.get_fnu0()
    lums = _spec.get_broadband_luminosities(fc)
    mags = [lnu_to_absolute_mag(v) for k, v in lums.items()]
    # bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1])/2
    # h = 0.68

    # phi = calc_df(mags[2], (100/h)**3, binLimits)[0]
    # phi[phi == 0.] = 1.e-10
    # phi_r = np.log10(phi)

    g_r = mags[1] - mags[2]
    g_r = np.histogram(g_r, binLimits)[0]

    return torch.tensor(g_r, dtype=torch.float32)
    # return torch.tensor(np.hstack([phi_r, g_r]), dtype=torch.float32)


# num_dim = 3 # 2
prior = ili.utils.Uniform(
    low=torch.tensor([0.01, 0.3, 0.5]),
    high=torch.tensor([0.5, 1.5, 2.0]),
)

# num_dim = 5
# prior = utils.BoxUniform(low=[0.01, 0.5, 0.1, 0, 0.01], high=[0.5, 2.0, 0.3, 1, 0.05])

# specs = young_specs + old_specs
# apply_dust_partial = partial(apply_dust, specs=specs, fc=fc, binLimits=binLimits)
apply_dust_partial = partial(
    apply_dust,
    young_specs=young_specs,
    old_specs=old_specs,
    fc=fc,
    binLimits=binLimits
)
simulator, prior = prepare_for_sbi(apply_dust_partial, prior)

theta, x = simulate_for_sbi(
    simulator,
    proposal=prior,
    num_simulations=1000,
    num_workers=28,
)
loader = ili.dataloaders.NumpyLoader(x=x, theta=theta)


plt.plot(binLimits[:-1], x.squeeze().T, color='k', alpha=0.1)
# plt.show()
plt.savefig('test.png')
plt.close()

# np.savetxt('data/simulations.txt', np.hstack([theta, x]))
# dat = torch.tensor(np.loadtxt('data/simulations.txt'), dtype=torch.float32)
# theta, x = dat[:,:2], dat[:,2:]

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

theta_o = np.random.uniform(low=[0.01, 0.3, 0.5], high=[0.5, 1.5, 2.0])
# x_o = apply_dust(theta_o, specs, fc, binLimits)
x_o = apply_dust(theta_o, young_specs, old_specs, fc, binLimits)


x_o += np.random.normal(0, scale=100, size=len(x_o)).astype(int)
x_o[x_o < 0] = 0.


"""
get posterior samples
"""
posterior_samples = posterior.sample((10000,), x=x_o)
posterior_samples = posterior_samples.detach().cpu().numpy()


"""
plot corner
"""
fig = corner.corner(
    posterior_samples,
    # labels=['tau_v', 'slope'],
    labels=['tau_v_ism', 'tau_v_bc', 'slope'],
    quantiles=[0.16, 0.5, 0.84],  # 0.5
    show_titles=True,
    title_kwargs={"fontsize": 12},
    plot_datapoints=False
)

corner.overplot_lines(fig, theta_o, color="C1")
corner.overplot_points(fig, theta_o[None], marker="s", color="C1")

ax = fig.add_axes([0.6, 0.6, 0.3, 0.3])

bins = binLimits[:-1] + (binLimits[1:] - binLimits[:-1])/2
# with np.errstate(divide='ignore'):
# , device='cuda')
mean = torch.tensor(np.median(np.array(posterior_samples), axis=0))
# ax.plot(bins, apply_dust(mean, specs, fc, binLimits), label='Posterior median', zorder=2)
ax.plot(bins, apply_dust(mean, young_specs, old_specs, fc,
        binLimits), label='Posterior median', zorder=2)
ax.plot(bins, x_o, label='True', zorder=3)

# with MultiPool() as pool:
# [ax.plot(bins, apply_dust(posterior_samples[i], specs, fc, binLimits),
[ax.plot(bins, apply_dust(posterior_samples[i], young_specs, old_specs, fc, binLimits),
         alpha=0.1, color='black', zorder=0) for i in np.arange(10)]

# ax.set_ylim(-4.5, -0.5);
# ax.set_xlim(10, 14.4)
ax.grid(alpha=0.1)
# ax.set_xlabel('$M_{\mathrm{halo}} \,/\, \mathrm{M_{\odot}}$')
ax.set_ylabel('$\phi \,/\, \mathrm{Mpc^{-3} \; dex^{-1}}$')
ax.legend()

plt.show()
# plt.savefig(f'sbi_example.png', dpi=250); plt.close()
