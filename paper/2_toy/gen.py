import os
import numpy as np


def simulator(params):
    # create toy simulations
    x = np.arange(0, 10)
    y = 3*np.sin(x+params[0]) + (params[1]-(params[2]-0.5)**2+0.5) * x ** 2
    y += np.random.randn(len(x))
    return y


if __name__ == '__main__':
    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate data and save as numpy files
    theta = np.random.randn(2000, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)
