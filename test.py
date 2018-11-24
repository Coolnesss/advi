import data
import advi
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np 
from torch import distributions
from torch.distributions import Bernoulli, Normal, MultivariateNormal
from torch import Tensor

import pyro_trainer

def log_reg_test():
    X, y = data.create_logreg_dataset()

    def posterior_func(theta):
        return Bernoulli(torch.sigmoid(theta.squeeze() @ X.t())).log_prob(y).sum() + Normal(0, 1).log_prob(theta).sum()
    
    print("asd",posterior_func(torch.randn(9)).shape)
    m = advi.ADVI()
    m.learn(posterior_func, n_params=9)


def gaussian_test():
    n_samples = 1000
    data = Normal(5, 1).sample((n_samples, ))

    def posterior_func(theta):
        return Normal(theta, 1).log_prob(data).sum() + Normal(5, 5).log_prob(theta).sum()
    
    m = advi.ADVI()
    m.learn(posterior_func, n_params=1)

def two_dim_gaussian_test():
    # Generate a skewed 2D gaussian distribution
    n_samples = 100
    sigma = Tensor([
        [1,     0],
        [0.9,   1.0]])
    data = MultivariateNormal(Tensor([0,0]), covariance_matrix=sigma).sample((n_samples, ))

    #plt.scatter(data[:,0].numpy(), data[:,1].numpy(), s=3)
    #plt.xlim(-4, 4)
    #plt.ylim(-4, 4)
    #plt.show()

    pyro_trainer.train_gaussian(data)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)
    #gaussian_test()
    two_dim_gaussian_test()
