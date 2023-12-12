import os
import numpy as np
import torch


def simulator(params):
    # create toy simulations
    params = torch.atleast_2d(params)
    x = torch.arange(1, 11)
    y = 2*torch.sin(torch.outer(params[:, 0], x))
    y += 10 * torch.outer(params[:, 1] * torch.sqrt(params[:, 2]), 1/(x**2))
    y += torch.randn(len(x))
    return y


if __name__ == '__main__':
    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate data and save as numpy files
    theta = torch.rand(1000, 3)  # 200 simulations, 3 parameters
    x = simulator(theta)
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)

    theta_obs = torch.Tensor([[0.412, 0.63, 0.39]])
    x_obs = simulator(theta_obs)
    print(theta_obs)
    np.save("toy/theta_obs.npy", theta_obs)
    np.save("toy/x_obs.npy", x_obs)
