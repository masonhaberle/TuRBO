#!/usr/bin/env python3
import torch
import numpy as np

from gpytorch.kernels.rbf_kernel import RBFKernel, postprocess_rbf


class RBFKernelDirectionalGrad(RBFKernel):
    r"""
    Pass in v1 and v2 through the params. If v1 has n_dir1 directions per
    point in x2 then it should be shape n1*n_dir1 x dim. The directions
    are assumed to be stored in blocks so that the first n_dir1 directions
    belong to x1[0] and the second n_dir1 directions belong to x1[1] etc.
    
    If you have a single set of global directions such as torch.eye(dim), then 
    you can repeat those to make v1 and v2 with 
    v1 = torch.eye(dim).repeat(n1,1)

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.


    """

    def __init__(self, ard_num_dims=None, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        v1 = params['v1']
        v2 = params['v2']
        # number of directions per point
        n_dir1 = int(v1.shape[-2]/n1)
        n_dir2 = int(v2.shape[-2]/n2)

        # set num the number of directions for num_outputs_per_input
        self.set_num_directions(n_dir1,n_dir2)

        # normalize directions
        v1 = (v1.T/torch.norm(v1,dim=1)).T
        v2 = (v2.T/torch.norm(v2,dim=1)).T

        # ARD: scale both inputs and directions by lengthscale
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        v1_ = v1 / self.lengthscale
        v2_ = v2 / self.lengthscale

        K = torch.zeros(*batch_shape, n1 * (n_dir1 + 1), n2 * (n_dir2 + 1), device=x1.device, dtype=x1.dtype)
        K = torch.zeros(*batch_shape, n1 * (n_dir1 + 1), n2 * (n_dir2 + 1), device=x1.device, dtype=x1.dtype)

        if not diag:
            # 1) Kernel block
            sq_dist = self.covar_dist(x1_, x2_, square_dist=True)
            K_11 = torch.exp(-0.5 * sq_dist)
            # print("K_11 block (function-function):\n", K_11.detach().numpy())
            # print("K_11 diagonal:", K_11.diag().detach().numpy())
            K[..., :n1, :n2] = K_11

            # 2) First gradient block
            x2_v2 = x2_.reshape(n2,1,d).bmm(torch.transpose(v2_.reshape(n2,n_dir2,d),-2,-1))
            x1_v2 = x1_ @ v2_.T
            outer  = x1_v2 - x2_v2.flatten()
            pi1 = torch.arange(n2 * (n_dir2)).view(n2,n_dir2).t().reshape((n2 * (n_dir2)))
            outer1 = outer[:,pi1]
            K[..., :n1, n2:] = outer1 * K_11.repeat([*([1] * (n_batch_dims + 1)), n_dir2]) 

            # Second gradient block
            x1_v1 = x1_.reshape(n1,1,d).bmm(torch.transpose(v1_.reshape(n1,n_dir1,d),-2,-1))
            x2_v1 = x2_ @ v1_.T
            outer  = x1_v1.flatten() - x2_v1
            pi2 = torch.arange(n1 * (n_dir1)).view(n1,n_dir1).t().reshape((n1 * (n_dir1)))
            outer2 = outer[:,pi2]
            outer2  = outer2.t()
            K[..., n1:, :n2] = -outer2 * K_11.repeat([n_dir1,*([1] * (n_batch_dims + 1))]) 

            # 4) Hessian block (n1*n_dir1, n2*n_dir2)
            outer3 = outer1.repeat(1, n_dir1, 1) * outer2.repeat(1,1,n_dir2)  
            kp = v1_ @ v2_.T
            kp = kp[:,pi1][pi2,:]
            chain_rule = kp - outer3
            K[..., n1:, n2:] = chain_rule * K_11.repeat([*([1] * n_batch_dims), n_dir1,n_dir2])
            
            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(n1 * (n_dir1 + 1)).view(n_dir1 + 1, n1).t().reshape((n1 * (n_dir1 + 1)))
            pi2 = torch.arange(n2 * (n_dir2 + 1)).view(n_dir2 + 1, n2).t().reshape((n2 * (n_dir2 + 1)))
            K = K[..., pi1, :][..., :, pi2]
            return K

        else:
            if not (n1 == n2 and torch.eq(x1, x2).all() and n_dir1 == n_dir2 and torch.eq(v1, v2).all()):
                raise RuntimeError("diag=True only works when x1 == x2 and v1 == v2")

            kernel_diag = super(RBFKernelDirectionalGrad, self).forward(x1, x2, diag=True)
            # Compute the variance for each directional derivative
            v2 = params['v2']  # shape: (n2 * n_dir2, d)
            lengthscale = self.lengthscale.view(1, -1)  # shape: (1, d)
            v2_scaled = v2 / lengthscale  # shape: (n2 * n_dir2, d)
            norm_sq = (v2_scaled ** 2).sum(dim=1)  # shape: (n2 * n_dir2,)
            grad_diag = norm_sq.view(n2, n_dir2)
            kernel_diag = kernel_diag.view(n2, 1)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            return k_diag.reshape(-1)

    def set_num_directions(self,n_dir1,n_dir2):
        """needed num_outputs_per_intput doesnt take v1,v2 as 
           args"""
        self.n_dir1 = n_dir1
        self.n_dir2 = n_dir2

    def num_outputs_per_input(self, x1, x2):
        return (self.n_dir1 +1,self.n_dir2 +1)
        # return self.n_dir1+1



if __name__ == '__main__':

  torch.manual_seed(0)
  # generate training data
  n1   = 2
  n2   = n1
  dim = 2
  train_x  = torch.rand(n1,dim)
  train_x2 = train_x
  # set directions
  n_directions = 2
  v1 = torch.eye(dim).repeat(n1,1)
  v2 = v1

  v1 = (v1.T/torch.norm(v1,dim=1)).T
  v2 = (v2.T/torch.norm(v2,dim=1)).T

  k = RBFKernelDirectionalGrad(ard_num_dims=dim)
  params = {'v1':v1,'v2':v2}
  K = k(train_x,train_x2, **params)
  print("Custom kernel matrix:\n", K.detach().numpy())

  # Print custom kernel parameters
  print("Custom kernel parameters:")
  print("  lengthscale:", k.lengthscale)
  if hasattr(k, 'outputscale'):
      print("  outputscale:", k.outputscale)

  # (1) Compare to GPyTorch's RBFKernelGrad if available
  try:
      from gpytorch.kernels import RBFKernelGrad
      k_builtin = RBFKernelGrad(ard_num_dims=dim)
      print("GPyTorch RBFKernelGrad parameters:")
      print("  lengthscale:", k_builtin.lengthscale)
      if hasattr(k_builtin, 'outputscale'):
          print("  outputscale:", k_builtin.outputscale)
      K_builtin = k_builtin(train_x, train_x2)
      K_builtin_dense = K_builtin.to_dense()
      print("Kernel matrix:\n", K_builtin_dense.detach().numpy())

      diff = (K_builtin_dense - K).to_dense().abs().max()
      print("Max abs difference with GPyTorch RBFKernelGrad:", diff.item())
  except Exception as e:
      print("Could not compare to GPyTorch RBFKernelGrad:", e)

  # (2) Check direction redundancy
  directions = v1.cpu().numpy()
  rank = np.linalg.matrix_rank(directions)
  print("Directions shape:", directions.shape, "Rank:", rank)

  # (5) Print kernel hyperparameters
  print("Lengthscales:", k.lengthscale)
  if hasattr(k, 'outputscale'):
      print("Outputscale:", k.outputscale)
  # If you have a likelihood, print its noise (example only)
  try:
      from gpytorch.likelihoods import GaussianLikelihood
      likelihood = GaussianLikelihood()
      print("Noise:", likelihood.noise)
  except Exception as e:
      print("Could not print likelihood noise:", e)
