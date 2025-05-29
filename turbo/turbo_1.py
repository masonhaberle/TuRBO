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
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")
    initial_lengthscales : Initial lengthscales for GP initialization, numpy.array, shape (d,)

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        scale=1.0,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        initial_lengthscales=None,
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"
        if initial_lengthscales is not None:
            assert len(initial_lengthscales) == len(lb), "initial_lengthscales must match dimension"
            assert np.all(initial_lengthscales > 0), "initial_lengthscales must be positive"
        else:
            self.initial_lengthscales = np.ones(len(lb))

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub
        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.initial_lengthscales = initial_lengthscales

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(200 * self.dim, 5000)  # Increased candidate points for better coverage
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6 * 8
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

        if self.verbose:
            print(f"Initial trust region: length_init={self.length_init}, length_min={self.length_min}, length_max={self.length_max}")
            sys.stdout.flush()

        # Shrinking parameters
        self.no_improve_count = 0
        self.shrink_patience = 20  # Number of iterations to wait before shrinking
        self.shrink_factor = 0.5   # Fraction to shrink the box by
        self.shrink_threshold = 0.1  # Minimum improvement to reset counter
        self.last_best_f = np.inf

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        prev_length = self.length
        log_fX_next = np.log1p(fX_next)
        log_fX = np.log1p(self._fX)
        # print(np.min(log_fX_next), np.min(log_fX), math.fabs(np.min(log_fX)))
        if np.min(log_fX_next) < np.min(log_fX) - 1e-3 * math.fabs(np.min(log_fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            if self.verbose:
                print(f"[TR EXPAND] Trust region expanded: {prev_length} -> {self.length}")
                sys.stdout.flush()
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            if self.verbose:
                print(f"[TR CONTRACT] Trust region contracted: {prev_length} -> {self.length}")
                sys.stdout.flush()
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Transform to log space and standardize
        log_fX = np.log(fX)
        log_mu = np.median(log_fX)
        log_sigma = np.std(log_fX)
        log_sigma = 1.0 if log_sigma < 1e-6 else log_sigma
        fX_copy = (log_fX - log_mu) / log_sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX_copy).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, 
                initial_lengthscales=self.initial_lengthscales,
                use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # GP accuracy check (train MSE, lengthscales, noise)
        # if self.verbose:
        #     with torch.no_grad():
        #         y_pred = gp(X_torch).mean.cpu().numpy().ravel()
        #         train_mse = np.mean((y_pred - fX_copy.ravel()) ** 2)
        #         l = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        #         print(f"[GP CHECK] Train MSE: {train_mse:.4g}")
        #         sys.stdout.flush()

        # Create the trust region boundaries
        x_center = X[fX_copy.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)
        # if self.verbose:
            # print(f"[TR BOUNDS] length={length}, weights={weights}, width={ub-lb}")
            # print("lb : ", lb)
            # print("ub : ", ub)
            # sys.stdout.flush()

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(5.0 / self.dim, 1.0)  # Increased perturbation probability
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points with more exploration
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]
        
        # Add some random exploration points
        # n_random = int(0.1 * self.n_cand)  # 10% random points
        # random_points = np.random.rand(n_random, self.dim)
        # X_cand = np.vstack((X_cand, random_points))

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # Transform back from log space
        y_cand = np.expm1(y_cand * log_sigma + log_mu)

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
        return X_next

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                print(f"Starting from fbest = {fbest:.4}, log(fbest) = {np.log(fbest):.4}")
                sys.stdout.flush()

            # Initialize last_best_f for shrinking logic
            self.last_best_f = self._fX.min()
            self.no_improve_count = 0

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, 
                    n_training_steps=self.n_training_steps, hypers={}
                )

                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])

                # Expand the GLOBAL trust region bounds if any X_next is 
                # at/near the bounds
                # expand_threshold = 1e-4 * (self.ub - self.lb)  # 1e-2 to be loose
                # expand_amount = 0.1 * (self.ub - self.lb)
                # expanded = np.zeros(self.dim, dtype=bool)
                # for d in range(self.dim):
                #     if np.any(X_next[:, d] >= self.ub[d] - expand_threshold[d]):
                #         self.ub[d] += expand_amount[d]
                #         expanded[d] = True
                #         if self.verbose:
                #             print(f"[GLOBAL BOUND] Expanded upper bound in dim {d} to {self.ub[d]}")
                #             sys.stdout.flush()
                #     if np.any(X_next[:, d] <= self.lb[d] + expand_threshold[d]):
                #         self.lb[d] -= expand_amount[d]
                #         expanded[d] = True
                #         if self.verbose:
                #             print(f"[GLOBAL BOUND] Expanded lower bound in dim {d} to {self.lb[d]}")
                #             sys.stdout.flush()

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    n_evals, fbest = self.n_evals, fX_next.min()
                    print(f"{n_evals}) New best: {fbest:.4}, log(fbest) = {np.log(fbest):.4}")
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))

                # --- Shrink global bounds if no significant improvement ---
                # current_best_f = self.fX.min()
                # if current_best_f < self.last_best_f - self.shrink_threshold:
                #     self.no_improve_count = 0
                #     self.last_best_f = current_best_f
                # else:
                #     self.no_improve_count += 1

                # if self.no_improve_count >= self.shrink_patience:
                #     # Shrink bounds around current best point
                #     best_idx = np.argmin(self.fX)
                #     best_x = self.X[best_idx]
                #     box_size = self.shrink_factor * (self.ub - self.lb)
                #     new_lb = np.maximum(best_x - box_size / 2, self.lb)
                #     new_ub = np.minimum(best_x + box_size / 2, self.ub)
                #     if self.verbose:
                #         print(f"[GLOBAL BOUND] Shrinking bounds around best_x. New lb: {new_lb}, New ub: {new_ub}")
                #         sys.stdout.flush()
                #     self.lb = new_lb
                #     self.ub = new_ub
                #     self.no_improve_count = 0
