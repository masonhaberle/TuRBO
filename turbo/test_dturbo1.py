import numpy as np
import torch
from dturbo1 import *

def myObj(u):
  # stack it
  fg = np.zeros(len(u)+1)
  fg[0] = 10*u @ u / 2
  fg[1:] = np.copy(10*u)
  return fg

# Set TuRBO Params
dim = 10
if torch.cuda.is_available():
  turbo_device = 'cuda'
else:
  turbo_device = 'cpu'

# initialize TuRBO
problem = DTurbo1(
      myObj,
      lb=-np.ones(dim),
      ub=np.ones(dim),
      n_init=50,
      max_evals=150,
      batch_size=5,
      verbose=True,
      use_ard=True,
      max_cholesky_size=2000,
      num_inducing=200,
      num_directions=1,
      n_training_steps=100,
      min_cuda=0, # directional_vi.py always runs on cuda if available
      device=turbo_device,
      dtype="float64")

# optimize
problem.optimize()
X_turbo, fX_turbo = problem.X, problem.fX[:,0] # Evaluated points

# get the optimum
idx_opt = np.argmin(fX_turbo)
fopt = fX_turbo[idx_opt]
xopt = X_turbo[idx_opt]
print(f"fopt = {fopt}")
