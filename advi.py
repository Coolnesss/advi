import torch
from torch.autograd import Variable


class ADVI:
    def __init__(self, M=1, ):
        self.M = M

    def learn(self, posterior_func, n_params=5, q_dist="mean"):

        mu = Variable(torch.zeros(n_params), requires_grad=True)
        omega = Variable(torch.ones(n_params), requires_grad=True)
        
        print("starting mu and omega", mu, omega)

        for i in range(3000):
            epsilon = torch.randn(n_params)
            xi = (1.0 / omega) * (epsilon - mu)

            theta = Variable(xi, requires_grad=True) # TODO replace with real transformation
            print("theta",theta)
            posterior = posterior_func(theta)
            posterior.backward()

            mu_gradient = theta.grad
            
            # note formula, GRADIENT is given in eq 8!
            omega_gradient = (mu_gradient * (epsilon @ torch.diag(torch.exp(omega)))) + 1

            mu = mu + torch.tensor([1e-4]) * mu_gradient
            omega = omega + torch.tensor([1e-4]) * omega_gradient

            theta.grad.zero_()
            assert theta.grad == 0

            if i % 500 == 0:
                print("mu now", mu, "sigma", omega)

        print("ended with", mu)


