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
from torch.utils.data import TensorDataset, DataLoader

from .utils import from_unit_cube, latin_hypercube, to_unit_cube
from .dsvgp import train_gp as train_gp_derivative

class DTurbo1:
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
    num_inducing : number of inducing points
    num_directions :  number of inducing directions per point
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
        df,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=10,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        num_inducing=500,
        num_directions=1,
        n_training_steps=150,
        minibatch_size=20,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
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

        # Save function information
        self.f = f
        self.df = df
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

        # DSVGP settings
        self.num_inducing=num_inducing
        self.num_directions=num_directions
        self.n_training_steps=n_training_steps
        self.minibatch_size=minibatch_size
        self.lr_hypers=0.01 # LR, 0.01 or 0.1
        self.lr_sched=None  # No LR schedule
        self.mll_type="PLL" # PPGPR

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 500)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6 
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, self.dim + 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def train_gp(self,train_x, train_y):
        """make a trainable model for DTuRBO
        Wraps DSVGP training module
        Expects train_x on unit cube and train_y standardized
        """
        train_x = train_x.float()
        train_y = train_y.float()
        dataset = TensorDataset(train_x,train_y)

        # train
        model,likelihood = train_gp_derivative(dataset,
                            num_inducing=self.num_inducing,
                            num_directions=self.num_directions,
                            minibatch_size = self.minibatch_size,
                            minibatch_dim = self.num_directions,
                            num_epochs =self.n_training_steps, 
                            learning_rate_hypers=self.lr_hypers,
                            lr_sched=self.lr_sched,
                            mll_type=self.mll_type,
                            verbose=self.verbose,
                            )
        # number type is important
        return model.double(),likelihood.double()
    
    def sample_from_gp(self,model,likelihood,X_cand,n_samples):
        """Sample from the GP model
        X_cand: 2d torch tensor, points to sample at
        n_samples: int, number of samples to take per point in X_cand
        """
        model.eval()
        likelihood.eval()
    
        # ensure correct type
        model = model.float()
        likelihood = likelihood.float()
        X_cand = X_cand.float()
        
        n,dim = X_cand.shape
        kwargs = {}
        derivative_directions = torch.eye(dim)[:dim] # predict all partial derivs
        derivative_directions = derivative_directions.repeat(n,1)
        kwargs['derivative_directions'] = derivative_directions.to(X_cand.device).float()

        batch_size = 50  # Set batch size for candidate evaluation
        y_cand_list = []
        for i in range(0, n, batch_size):
            X_batch = X_cand[i:i+batch_size]
            d_batch = torch.eye(dim)[:dim].repeat(X_batch.shape[0],1).to(X_cand.device).float()
            batch_kwargs = {'derivative_directions': d_batch}
            preds = likelihood(model(X_batch, **batch_kwargs))
            y_batch = preds.sample(torch.Size([n_samples]))  # shape (n_samples, batch_size*(dim+1))
            y_batch = y_batch[:,::dim+1].t()  # shape (batch_size, n_samples)
            y_cand_list.append(y_batch.cpu())
        y_cand = torch.cat(y_cand_list, dim=0)
        return y_cand

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX[:,0]) - 1e-3 * math.fabs(np.min(self._fX[:,0])):
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
        #assert X.min() >= 0.0 and X.max() <= 1.0   # MP: turned off b/c model improv

        # Standardize function values.
        mu, sigma = np.median(fX, axis=0)[0], fX.std(axis=0)[0]
        fX[:,0] = (deepcopy(fX[:,0]) - mu) / sigma
        # Standardize gradients
        fX[:,1:] = deepcopy(fX[:,1:]) / sigma
        # do from_unit_cube mapping on gradients (b/c X got mapped to unit cube)
        fX[:,1:] = deepcopy(fX[:,1:]) * (self.ub-self.lb)

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp,likelihood = self.train_gp(
                train_x=X_torch, train_y=y_torch)

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX[:, 0].argmin().item(), :][None, :]
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
            # predict function values (self.n_cand, self.batch_size)
            y_cand = self.sample_from_gp(gp,likelihood,X_cand_torch,self.batch_size).cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled function values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, x_center

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        y_next = np.ones(self.batch_size)
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_next[i] = y_cand[indbest,i]
            y_cand[indbest, :] = np.inf
        return X_next,y_next


    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX[:, 0].min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.vstack([self.f(x) for x in X_init])
            dX_init = np.vstack([self.df(x) for x in X_init])
            fX_init = np.hstack([fX_init, dX_init])

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX[:, 0].min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

                # Standardize values
                fX = deepcopy(self._fX)

                # Create th next batch
                X_cand, y_cand, x_center = self._create_candidates(
                    X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next,y_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.vstack([self.f(x) for x in X_next])
                dX_next = np.vstack([self.df(x) for x in X_next])
                fX_next = np.hstack([fX_next, dX_next])

                # Expand the GLOBAL trust region bounds if any X_next is 
                # at/near the bounds
                # This is NOT in the original TuRBO paper but allows the algorithm
                # to adapt to potentially poor initialization of the global 
                # trust region bounds.
                expand_threshold = 1e-4 * (self.ub - self.lb)  # 1e-2 to be loose
                expand_amount = 0.1 * (self.ub - self.lb)
                expanded = np.zeros(self.dim, dtype=bool)
                for d in range(self.dim):
                    if np.any(X_next[:, d] >= self.ub[d] - expand_threshold[d]):
                        self.ub[d] += expand_amount[d]
                        expanded[d] = True
                        if self.verbose:
                            print(f"[GLOBAL BOUND] Expanded upper bound in dim {d} to {self.ub[d]}")
                            sys.stdout.flush()
                    if np.any(X_next[:, d] <= self.lb[d] + expand_threshold[d]):
                        self.lb[d] -= expand_amount[d]
                        expanded[d] = True
                        if self.verbose:
                            print(f"[GLOBAL BOUND] Expanded lower bound in dim {d} to {self.lb[d]}")
                            sys.stdout.flush()

                # Update trust region (based only on function values)
                self._adjust_length(fX_next[:,0])

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next[:, 0].min() < self.fX[:, 0].min():
                    n_evals, fbest = self.n_evals, fX_next[:, 0].min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))


