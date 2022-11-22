import numpy as np
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner

# create toy 'simulations'
theta = np.random.rand(2000, 3) # 2000 simulations, 3 parameters
def simulator(params):
    x = np.arange(10)
    y = params[0]*np.sin(x) + params[1]*x**2 - 3*params[2]*x
    y += np.random.normal(len(x))
    return y
x = np.array([simulator(t) for t in theta])

np.save('toy/theta.npy', theta)
np.save('toy/x.npy', x)

runner = SBIRunner.from_config('configs/sample_inference_config.yaml')

runner()

valrunner = ValidationRunner.from_config('configs/sample_validation_config.yaml')
valrunner()
