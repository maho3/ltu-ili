
from os.path import join
import numpy as np
import pymc3 as pm


def simulator(params):
    # create toy simulations
    x = np.arange(1, 11)
    y = 2*np.sin(x*params[0]) + 10*(params[1] * np.sqrt(params[2]))/(x**2)
    # y += np.random.randn(len(x))
    return y


wdir = '/home/mattho/git/ltu-ili/paper/wdir/toy'
xobs = np.load(join(wdir, 'x_obs.npy'))
thetaobs = np.load(join(wdir, 'theta_obs.npy'))[0]
ndim = len(thetaobs)
print(thetaobs)

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    theta = pm.Uniform("theta",
                       lower=[0]*ndim, upper=[1]*ndim,
                       shape=ndim)
    y_ = simulator(theta)

    # Likelihood (sampling distribution) of observations
    Y = pm.Normal("Y", mu=y_, sigma=1, observed=xobs)

with basic_model:
    # draw 500 posterior samples
    idata = pm.sample(10000, chains=12, cores=12, tune=20000,
                      step=pm.NUTS(),
                      progressbar=True, discard_tuned_samples=True)

# Save samples
samples = idata['theta', ::10]
print(samples.shape)
np.save(join(wdir, 'hmc_samples.npy'), samples)
