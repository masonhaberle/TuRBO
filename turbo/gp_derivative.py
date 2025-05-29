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
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.variational import CholeskyVariationalDistribution
from .GradVariationalStrategy import GradVariationalStrategy
from gpytorch.means import ConstantMeanGrad
from gpytorch.kernels import RBFKernelGrad
from gpytorch.distributions import MultitaskMultivariateNormal


# GP Model
class GPModelWithDerivatives(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, hypers={}):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 2
    assert train_x.shape[0] == train_y.shape[0]
    print(train_x.shape, train_y.shape)

    # Create hyper parameter bounds
    noise_constraint = Interval(1e-5, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.001, 5)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 200.0)  # Reduce lower bound to 0.005?

    # Create models
    likelihood = MultitaskGaussianLikelihood(
        num_tasks=2, 
        noise_constraint=noise_constraint
        ).to(device=train_x.device, dtype=train_y.dtype)
    ard_dims = train_x.shape[1] if use_ard else None
    model = GPModelWithDerivatives(
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
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.05
        model.initialize(**hypers)

    # Use the adam optimizer
    # optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=1e-1)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=1e-1)
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
        
        print('Iter: ', i + 1, ', Loss: ', loss.item())
        # print('outputscale: ', model.covar_module.outputscale.item())
        # print('lengthscales: ', model.covar_module.base_kernel.lengthscale, ', noise: ', model.likelihood.noise.item())
        optimizer.step()

    l = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    print("ℓ min, max:",   np.min(l), np.max(l))
    print("σ²:",  model.covar_module.outputscale.detach().cpu())
    print("η²:",  model.likelihood.noise_covar.noise.detach().cpu())

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    return model


class GPModelWithDerivativesSVI(gpytorch.models.ApproximateGP):
    """
    Sparse variational GP for function + gradient observations using GradVariationalStrategy.
    Uses RBFKernelGrad and ConstantMeanGrad. Suitable for large-scale problems with derivatives.
    """
    def __init__(self, inducing_points, input_dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = GradVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMeanGrad()
        self.base_kernel = RBFKernelGrad(ard_num_dims=input_dim)
        self.covar_module = ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


def train_gp_svi(train_x, train_y, num_inducing=20, num_steps=100, lr=0.01, verbose=False):
    """
    Train a sparse variational GP with derivatives using GradVariationalStrategy.
    Args:
        train_x: torch.Tensor, shape [N, d]
        train_y: torch.Tensor, shape [N, d+1] (function value + gradients)
        num_inducing: int, number of inducing points
        num_steps: int, number of SVI steps
        lr: float, learning rate
        verbose: bool, print loss during training
    Returns:
        model, likelihood (both in eval mode)
    """
    N, d = train_x.shape
    # Select inducing points randomly from training data
    inducing_idx = np.random.choice(N, min(num_inducing, N), replace=False)
    inducing_points = train_x[inducing_idx]
    model = GPModelWithDerivativesSVI(inducing_points, input_dim=d).to(train_x.device)
    likelihood = MultitaskGaussianLikelihood(num_tasks=d+1).to(train_x.device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)
    batch_size = min(64, N)
    for i in range(num_steps):
        perm = torch.randperm(N)
        total_loss = 0.0
        for j in range(0, N, batch_size):
            idx = perm[j:j+batch_size]
            x_batch = train_x[idx]
            y_batch = train_y[idx]
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        if verbose and (i % 10 == 0 or i == num_steps-1):
            print(f"SVI iter {i+1}/{num_steps}, loss: {total_loss/N:.4f}")
    model.eval()
    likelihood.eval()
    return model, likelihood
