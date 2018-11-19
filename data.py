from torch.utils.data import Dataset, TensorDataset
from torch.distributions import MultivariateNormal, Normal, Bernoulli
from torch import nn
import torch

def create_logreg_dataset(n_betas=9, n_samples=1000):
    # Draw betas (not specified how exactly?)
    betas = Normal(0,1).sample((n_betas,))
    intercept = 1.0

    # Draw x from prior
    data = Normal(0,1).sample((n_samples, n_betas))

    def likelihood(betas, intercept, x):
        return Bernoulli(probs=torch.sigmoid(intercept + x@betas))

    # sample targets from likelihood
    targets = likelihood(betas, intercept, data).sample((n_samples, ))
    print("real betas", betas)
    return data, targets

