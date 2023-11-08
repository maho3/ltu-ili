import os
import numpy as np


def simulator(params):
    # create toy simulations
    x = np.arange(10)
    y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
    y += np.random.randn(len(x))
    return y


if __name__ == '__main__':
    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate data and save as numpy files
    theta = np.random.rand(2000, 3)  # 200 simulations, 3 parameters
    x = np.array([simulator(t) for t in theta])
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)
