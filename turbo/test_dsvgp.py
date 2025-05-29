import math
import numpy as np
import torch
import gpytorch
import tqdm
import random
import time
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy
from dsvgp import train_gp, eval_gp

# data parameters
n   = 600
dim = 2
n_test = 1000

# training params
num_inducing = 200
num_directions = 1
minibatch_size = 200
num_epochs = 300

# seed
torch.random.manual_seed(0)
# use tqdm or just have print statements
tqdm = False
# use data to initialize inducing stuff
inducing_data_initialization = False
# use natural gradients and/or CIQ
use_ngd = False
use_ciq = False
num_contour_quadrature=15
# learning rate
learning_rate_hypers = 0.01
learning_rate_ngd    = 0.1
gamma  = 10.0
#levels = np.array([20,150,300])
#def lr_sched(epoch):
#  a = np.sum(levels > epoch)
#  return (1./gamma)**a
lr_sched = None

def f(x, deriv=True):
  # f(x) = sin(2pi(x**2+y**2)), df/dx = cos(2pi(x**2+y**2))4pi*x
  fx = torch.sin(2*np.pi*torch.sum(x**2,dim=1))
  gx = 4*np.pi*( torch.cos(2*np.pi*torch.sum(x**2,dim=1)) * x.T).T
  fx = fx.reshape(len(x),1)
  if deriv:
    return torch.cat([fx,gx],1)
  else:   
    return fx.squeeze(axis=1)

# training and testing data
train_x = torch.rand(n,dim)
test_x = torch.rand(n_test,dim)
train_y = f(train_x, deriv=True)
test_y = f(test_x, deriv=True)
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=False)

# train
print("\n\n---DirectionalGradVGP---")
print(f"Start training with {n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
args={"verbose":True}
t1 = time.time()	
model,likelihood = train_gp(train_dataset,
                      num_inducing=num_inducing,
                      num_directions=num_directions,
                      minibatch_size = minibatch_size,
                      minibatch_dim = dim,
                      num_epochs =num_epochs, 
                      learning_rate_hypers=learning_rate_hypers,
                      learning_rate_ngd=learning_rate_ngd,
                      inducing_data_initialization=inducing_data_initialization,
                      use_ngd = use_ngd,
                      use_ciq = use_ciq,
                      lr_sched=lr_sched,
                      num_contour_quadrature=num_contour_quadrature,
                      tqdm=tqdm,**args
                      )
t2 = time.time()	

# save the model
# torch.save(model.state_dict(), "../data/test_dvi_basic.model")

# test
means, variances = eval_gp( test_dataset,model,likelihood,
                            num_directions=num_directions,
                            minibatch_size=n_test)
t3 = time.time()	
print(means.shape)
# compute MSE
test_y = test_y.cpu()
test_mse = torch.mean((test_y[:,0]-means[::dim+1])**2)
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means[::dim+1], variances.sqrt()[::dim+1]).log_prob(test_y[:,0]).mean()
print(f"At {n_test} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")
print(f"Training time: {(t2-t1):.2f} sec, testing time: {(t3-t2):.2f} sec")

import matplotlib.pyplot as plt
plot=1
if plot == 1:
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y[:,0], color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means[::dim+1], color='b')
    plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
    plt.show()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y[:,1], color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means[1::dim+1], color='b')
    plt.title("df/dx variational fit; actual curve is black, variational is blue")
    plt.show()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y[:,2], color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means[2::dim+1], color='b')
    plt.title("df/dy variational fit; actual curve is black, variational is blue")
    plt.show()
