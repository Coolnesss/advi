import data
import advi
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np 
from torch import distributions
from torch.distributions import Bernoulli, Normal

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

if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)
    gaussian_test()