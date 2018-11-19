import pyro
import torch
from torch import nn
from pyro.distributions import Normal

# Model for pyro
class LogisticRegression(nn.Module):
    def __init__(self, n_coefs=9):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_coefs, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train_logreg(data, model):

    def pyro_model(data):
        loc, scale = torch.zeros(1, 1), torch.ones(1, 1)
        bias_loc, bias_scale = torch.zeros(1), torch.ones(1)

        w_prior = Normal(loc, scale).independent(1)
        b_prior = Normal(bias_loc, bias_scale).independent(1)
        
        priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
        lifted_module = pyro.random_module("module", model, priors)

        lifted_reg_model = lifted_module()
        with pyro.iarange("map", len(data)):
            x, y = data
            prediction_mean = lifted_reg_model(x).squeeze(-1)
            # condition on the observed data
            pyro.sample("obs",
                        Normal(prediction_mean, 0.1 * torch.ones(data.size(0))),
                        obs=y)

    def guide(data):
        # define our variational parameters
        w_loc = torch.randn(1, 1)
        # note that we initialize our scales to be pretty narrow
        w_log_sig = torch.tensor(-3.0 * torch.ones(1, 1) + 0.05 * torch.randn(1, 1))
        b_loc = torch.randn(1)
        b_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
        # register learnable params in the param store
        mw_param = pyro.param("guide_mean_weight", w_loc)
        sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
        mb_param = pyro.param("guide_mean_bias", b_loc)
        sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
        # guide distributions for w and b
        w_dist = Normal(mw_param, sw_param).independent(1)
        b_dist = Normal(mb_param, sb_param).independent(1)
        dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
        # overload the parameters in the module with random samples
        # from the guide distributions
        lifted_module = pyro.random_module("module", regression_model, dists)
        # sample a regressor (which also samples w and b)
        return lifted_module()

    