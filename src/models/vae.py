import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from realnvp import Flow, LatentMaskedAffineCoupling, NormalisingFlow, MLP

class VarEncoder(nn.Module):
    def __init__(self, latent_size=64, kernel_size=5, img_height=32, channels=3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        
        # Calculate the output size of the image tensor after 4 convolution layers
        out_h = img_height
        out_w = img_height
        for i in range(4):
            out_h, out_w = self.convOutputShape((out_h, out_w), kernel_size)
        
        self.flattened_size = out_h * out_w * 128
        self.latent_img_height = out_h
        
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flattened_size, latent_size, bias=True)
        )
        
        self.fc_sigma = nn.Sequential(
            nn.Linear(self.flattened_size, latent_size, bias=True)
        )
        
    def forward(self, x):
        z = self.conv(x).view(x.shape[0], -1)
        mu = self.fc_mu(z)
        sigma = torch.exp(self.fc_sigma(z)) + 1e-7
        return mu, sigma
    
    def convOutputShape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = math.floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = math.floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w
    

class VarDecoder(nn.Module):
    def __init__(self, flattened_size, latent_img_height, latent_size=64, kernel_size=5, channels=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, flattened_size)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size)
        )
        self.latent_img_height = latent_img_height
        
        
    def forward(self, x):
        z = self.fc(x).view(x.shape[0], 128, self.latent_img_height, self.latent_img_height)
        im = self.conv(z)
        return im


class VariationalAutoencoder(nn.Module):
    def __init__(self, device, flows = None, latent_size=64, kernel_size=5, img_height=32, channels=3):
        super().__init__()
        self.encoder = VarEncoder(latent_size, kernel_size, img_height, channels)
        self.decoder = VarDecoder(self.encoder.flattened_size, self.encoder.latent_img_height)
        self.device = device
        self.flows = flows
        self.latent_size = latent_size
        
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z0 = self.reparameterize(mu, sigma)
        
        if self.flows is None:
            zk = z0
            var_loss = -0.5 * torch.sum(1 + torch.log(sigma) - mu.pow(2) - sigma)
        else:
            log_pi = torch.log(torch.tensor(2 * np.pi))
            log_sigma = torch.log(sigma)
            
            assert(not torch.any(torch.isnan(log_sigma)))
            assert(not torch.any(torch.isnan(z0)))
            assert(not torch.any(torch.isnan(mu)))
            
            log_prob_z0 = torch.sum(
                -0.5 * log_pi - log_sigma - 0.5 * ((z0 - mu) / sigma) ** 2, 
                axis=1)
            
            zk, log_det = self.flows(z0)
            log_prob_zk = torch.sum(-0.5 * log_pi - 0.5 * (zk**2), axis=1)
            
            var_loss = torch.mean(log_prob_z0) - torch.mean(log_prob_zk) - torch.mean(log_det)
            
        im = self.decoder(zk)
        return im, mu, sigma, var_loss
    
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(mu, device=self.device)
        z = mu + sigma * epsilon
        return z
    
    def sample(self, batch_size):
        # Sample from N(0, 1)
        z = torch.randn((batch_size, self.latent_size), device=self.device)
        if self.flows is not None:
            z, _ = self.flows(z)
        im = self.decoder(z)
        return im