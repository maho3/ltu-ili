from os.path import join
import numpy as np
import sbibm

outdir = './slcp/data/'
print(f'Writing to {outdir}...')

task = sbibm.get_task("slcp")
prior = task.get_prior()
simulator = task.get_simulator()

# Generate data
print('Generating data...')
Ns = [1_000, 10_000, 100_000]

for n in Ns:
    print(f'Generating {n} samples...')

    thetas = prior(num_samples=n)
    xs = simulator(thetas)

    filename = join(outdir, f'{n}')
    print(f'Saving to {filename}...')
    np.save(filename+'_x.npy', xs)
    np.save(filename+'_theta.npy', thetas)

# Generate reference
print('Getting reference samples...')
samples = [
    task.get_reference_posterior_samples(num_observation=i)
    for i in range(1, 11)
]

samples = np.stack(samples, axis=0)

filename = join(outdir, 'ref.npy')
print(f'Saving to {filename}...')
np.save(filename, samples)

# Generate obs data
print('Generating observed data...')
observed = [
    task.get_observation(num_observation=i)
    for i in range(1, 11)
]

for i, obs in enumerate(observed):
    filename = join(outdir, f'obs_{i+1}.npy')
    print(f'Saving to {filename}...')
    np.save(filename, obs)

# faking true parameters
theta = np.zeros_like(thetas[0])
filename = join(outdir, 'obstrue.npy')
print(f'Saving to {filename}...')
np.save(filename, theta)
