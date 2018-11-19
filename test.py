import data
import advi
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np 
from torch import distributions
from torch.distributions import Bernoulli

def log_reg_test():
    X, y = data.create_logreg_dataset()

    def posterior_func(theta):
        return Bernoulli(torch.sigmoid(X @ theta.t())).log_prob(y)
    
    m = advi.ADVI()
    m.learn(posterior_func, n_params=9)
    
if __name__ == '__main__':

    log_reg_test()