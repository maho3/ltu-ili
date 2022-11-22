import numpy as np
from ili.inference.runner_sbi import SBIRunner

# create toy 'simulations'
theta = np.random.rand(2000, 2) # 2000 simulations, 2 parameters
def simulator(params):
    x = np.arange(10)
    y = params[0]*np.sin(x) + params[1]*x**2
    y += np.random.normal(len(x))
    return y
x = np.array([simulator(t) for t in theta])

np.save('toy/theta.npy', theta)
np.save('toy/x.npy', x)

runner = SBIRunner.from_config('configs/sample_inference_config.yaml')

runner()
