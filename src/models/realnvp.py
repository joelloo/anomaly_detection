import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np
import math


class Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError("Forward pass not implemented")
    
    def inverse(self, z):
        raise NotImplementedError("Inverse pass not implemented")


class LatentMaskedAffineCoupling(Flow):
    def __init__(self, b, net):
        super().__init__()
        self.register_buffer('b', b)
        self.net = net
        self.scaling = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        z_masked = self.b * z
        s, t = self.net(z_masked).chunk(2, dim=1)

        s_exp = self.scaling.exp()
        s = torch.tanh(s / s_exp) * s_exp

        z_out = z_masked + (1 - self.b) * (z * torch.exp(s) + t)
        log_det = torch.sum((1 - self.b) * s, dim=list(range(1, self.b.dim())))
        return z_out, log_det

    def inverse(self, z):
        z_masked = self.b * z
        s, t = self.net(z_masked).chunk(2, dim=1)
        
        s_exp = self.scaling.exp()
        s = torch.tanh(s / s_exp) * s_exp

        z_out = z_masked + (1 - self.b) * (z - t) * torch.exp(-s)
        log_det = -torch.sum((1 - self.b) * s, dim=list(range(1, self.b.dim())))
        return z_out, log_det


class NormalisingFlow(nn.Module):
    def __init__(self, flows, prior, device):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = prior # Target distribution to be approximated
        self.device = device
        
    def forward(self, z):
        log_det = torch.zeros(z.shape, device=self.device)
        for flow in self.flows:
            z, local_log_det = flow(z)
            log_det += local_log_det
        return z, log_det

    def sample(self, sample_shape=None):
        if sample_shape:
            z = self.prior.sample(sample_shape=sample_shape)
        else:
            z = self.prior.sample() # No need to specify for multivariate normal

        z = z.to(self.device)
        for flow in reversed(self.flows):
            z, _ = flow.inverse(z)
        return z
    
    def get_prior_log_prob(self, z):
        return self.prior.log_prob(z)


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers):
        super().__init__()
        if num_layers == 1:
            layers = [nn.Linear(in_size, 2*in_size)]
        else:
            layers = [nn.Linear(in_size, hidden_size, bias=True), nn.LeakyReLU()]
            for i in range(num_layers - 2):
                layers += [nn.Linear(hidden_size, hidden_size, bias=True), nn.LeakyReLU()]
            layers += [nn.Linear(hidden_size, 2*in_size, bias=True)]
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)