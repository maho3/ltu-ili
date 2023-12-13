import os
import numpy as np
import torch


def simulator(params):
    # create toy simulations
    params = torch.atleast_2d(params)
    x = torch.linspace(-3, 3, 10)
    y = 3 * \
        torch.sin(x+torch.outer(params[:, 0] +
                  params[:, 1], torch.ones(len(x))))
    y += torch.outer(params[:, 1] - 3*(params[:, 2])**2, x**2)
    y += torch.randn(len(x))
    return y


if __name__ == '__main__':
    # construct a working directory
    if not os.path.isdir("toy"):
        os.mkdir("toy")

    # simulate data and save as numpy files
    theta = torch.randn(1000, 3)  # 200 simulations, 3 parameters
    x = simulator(theta)
    np.save("toy/theta.npy", theta)
    np.save("toy/x.npy", x)

    theta_obs = torch.Tensor([[0.412, 0.67, 0.39]])
    x_obs = simulator(theta_obs).numpy()
    theta_obs = theta_obs.numpy()
    print(theta_obs)
    np.save("toy/theta_obs.npy", theta_obs)
    np.save("toy/x_obs.npy", x_obs)
