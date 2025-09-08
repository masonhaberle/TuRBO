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
import time

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
        max_evals=-1,
        objective_tol=-1,
        improvement_tol=-1,
        max_time=-1,
        batch_size=1,
        scale=1.0,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64"
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert isinstance(max_evals, int)
        assert max_evals > 0 or objective_tol > 0 or improvement_tol > 0 or max_time > 0
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 0 and isinstance(n_training_steps, int)
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.objective_tol = objective_tol
        self.improvement_tol = improvement_tol
        self.max_time = max_time
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(200 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0
        self.f_best = np.inf
        self.improvement_err = self.improvement_tol + 2
        self.start_time = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6 * 8
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.bestX = np.zeros((0, self.dim))
        self.bestfX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

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

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next

    def check_tols(self):
        if self.max_evals >= 0 and self.max_evals < self.n_evals:
            print(f"Stop condition met: Number of evaluations {self.n_evals}")
            return False
        if self.objective_tol > self.f_best:
            print(f"Stop condition met: Objective value of {self.f_best:.4}")
            return False
        if len(self.bestfX) > 5:
            latest = self.bestfX[-5:]
            improvement = np.max(latest) - np.min(latest)
            if self.improvement_tol > improvement:
                print(f"Stop condition met: Improvement of {improvement:.4}")
                return False
        if self.max_time >= 0 and time.time() - self.start_time > self.max_time:
            print(f"Stop condition met: Ran for total time {time.time() - self.start_time:.4}")
            return False
        return True

    def optimize(self):
        """Run the full optimization process."""
        self.start_time = time.time()
        while self.check_tols():
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
            newBestInd = np.argmin(fX_init)
            newBestX = X_init[newBestInd]
            newBestfX = fX_init[newBestInd]
            self.bestX = np.vstack((self.bestX, deepcopy(newBestX)))
            self.bestfX = np.vstack((self.bestfX, deepcopy(newBestfX)))
            self.f_best = min(self.f_best, newBestfX[0])
            

            if self.verbose:
                print(f"Starting from fbest = {self.f_best:.4}")
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.check_tols() and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX).ravel()

                # Create the next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])
                newBestInd = np.argmin(fX_next)
                newBestX = X_next[newBestInd]
                newBestfX = fX_next[newBestInd]

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if newBestfX[0] < self.f_best:
                    self.f_best = newBestfX[0]
                    if self.verbose:
                        print(f"{self.n_evals}) New best: {self.f_best:.4}")

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))
                self.bestX = np.vstack((self.bestX, deepcopy(newBestX)))
                self.bestfX = np.vstack((self.bestfX, deepcopy(newBestfX)))
            