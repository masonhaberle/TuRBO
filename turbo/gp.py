###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP


# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, initial_lengthscales=None, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    noise_constraint = Interval(1e-5, 0.2)
    output_initialization = 1.0
    noise_initialization = 0.005
    lengthscales_initialization = 0.5
    if use_ard:
        if initial_lengthscales is not None:
            # Convert to numpy array first, then to torch tensor
            lengthscales_np = np.array(initial_lengthscales)
            lengthscales_tensor = torch.from_numpy(lengthscales_np)
            lengthscales_initialization = lengthscales_tensor
            lengthscale_constraint = Interval(
                float(torch.min(lengthscales_tensor)) * 0.1, 
                float(torch.max(lengthscales_tensor)) * 2.0)
        else:
            lengthscale_constraint = Interval(0.001, 5)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)  # Reduce lower bound to 0.005?

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = output_initialization
        hypers["covar_module.base_kernel.lengthscale"] = lengthscales_initialization
        hypers["likelihood.noise"] = noise_initialization
        model.initialize(**hypers)

    # Use the adam optimizer
    # optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=1e-1)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=2e-1)
    # optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=0.1, weight_decay=1e-4)

    for i in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if i == 0:
            print('Initial loss: ', loss.item())
        elif i == num_steps - 1:
            print('Final loss: ', loss.item())
        loss.backward()
        
        # print('Iter: ', i + 1, ', Loss: ', loss.item())
        # print('outputscale: ', model.covar_module.outputscale.item())
        # print('lengthscales: ', model.covar_module.base_kernel.lengthscale, ', noise: ', model.likelihood.noise.item())
        optimizer.step()

    # l = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    # print(l)
    # print("ℓ min, max:",   np.min(l), np.max(l))
    # print("σ²:",  model.covar_module.outputscale.detach().cpu())
    # print("η²:",  model.likelihood.noise_covar.noise.detach().cpu())

    # exit()
    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model
