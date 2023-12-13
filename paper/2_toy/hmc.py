
from os.path import join
import numpy as np
import pymc3 as pm
# from gen import simulator


def simulator(params):
    # create toy simulations
    x = np.linspace(-3, 3, 10)
    y = 3*np.sin(x+params[0]+params[1])
    y += (params[1] - 3*(params[2])**2)*x**2
    # y += np.random.randn(len(x))
    return y


# def simulator(params):
#     # create toy simulations
#     y = pm.math.stack([
#         params[0]**2*params[1],
#         params[0]*np.exp(params[1]),
#         np.sin(params[2]),
#         np.tanh(params[1])
#     ])
#     # y += 0.1*np.random.randn(*y.shape)
#     return y


if __name__ == '__main__':
    basedir = '/Users/maho/git/ltu-ili'
    wdir = basedir+'/paper/wdir/toy'
    xobs = np.load(join(wdir, 'x_obs.npy'))[0]
    thetaobs = np.load(join(wdir, 'theta_obs.npy'))[0]
    ndim = len(thetaobs)
    print(thetaobs, xobs)

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        theta = pm.Normal("theta",
                          mu=np.zeros(ndim), sigma=np.ones(ndim),
                          shape=ndim)
        y_ = simulator(theta)

        # Likelihood (sampling distribution) of observations
        Y = pm.Normal("Y", mu=y_, sigma=1, observed=xobs)

    with basic_model:
        # draw 500 posterior samples
        idata = pm.sample(10000, chains=8, cores=4, tune=10000,
                          step=pm.NUTS(target_accept=0.99),
                          return_inferencedata=False,
                          progressbar=True, discard_tuned_samples=True)

    # Save samples
    samples = idata['theta', ::10]
    print(samples.shape)
    np.save(join(wdir, 'hmc_samples.npy'), samples)
