import torch
from torch.autograd import Variable


class ADVI:
    def __init__(self, M=1, ):
        self.M = M
        pass

    def learn(self, posterior_func, n_params=5, q_dist="mean"):

        mu = Variable(torch.zeros(n_params), requires_grad=True)
        omega = Variable(torch.zeros(n_params), requires_grad=True)

        for i in range(1000):

            epsilon = torch.randn(self.M, n_params)
            xi = (1/omega) * (epsilon - mu)
            theta = xi # TODO replace with real transformation

            posterior = posterior_func(theta).sum(1)

            print("posterior mean", posterior.mean())
            mu_gradient = torch.autograd.grad(posterior.mean(), mu)
            (epsilon @ torch.diag(torch.exp(omega)))
            omega_grad_part = (posterior * (epsilon @ torch.diag(torch.exp(omega)))).mean() + 1
            omega_gradient = torch.autograd.grad(omega_grad_part, omega)


            mu += 1e-3 * mu_gradient
            omega += 1e-3 * omega_gradient

        print(mu)


