import sys
sys.path.append("..")
import numpy as np 
import random as rd
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
# from model.utils.gaussian_path_planner import GaussianRandomPathClass, kernel_levse, kernel_se, soft_squash
from utils.utils import np2torch, get_linear_layer


def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np
def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

class MLPG(nn.Module):
    def __init__(self, xdim=2, cdim=6, zdim=5, n_components=5, hdims=[128, 128], actv=nn.LeakyReLU(), actv_out=nn.Tanh(), spectral_norm=False, device='cpu'):
        super(MLPG, self).__init__()
        self.xdim     = xdim 
        self.cdim     = cdim 
        self.zdim     = zdim 
        self.n_components = n_components    
        self.actv     = actv
        self.actv_out = actv_out
        self.sn       = spectral_norm
        self.device   = device 

        decoder_hdims = list(reversed(hdims)).copy()
        decoder_hdims.insert(0, zdim+cdim)
        pi_layers    = get_linear_layer(decoder_hdims, nn.Tanh, initializer='normal')
        mu_layers    = get_linear_layer(decoder_hdims, nn.Tanh, initializer='normal')
        sigma_layers = get_linear_layer(decoder_hdims, nn.Tanh, initializer='normal')
        
        self.mdn_pi    = nn.Sequential(*pi_layers, nn.Linear(decoder_hdims[-1], n_components),  nn.Tanh())
        self.mdn_mu    = nn.Sequential(*mu_layers, nn.Linear(decoder_hdims[-1], n_components), nn.Tanh())
        self.mdn_sigma = nn.Sequential(*sigma_layers, nn.Linear(decoder_hdims[-1], n_components), nn.Sigmoid())

        encoder_hdims = hdims.copy()
        encoder_hdims.insert(0, xdim+cdim)
        encoder_layers = get_linear_layer(encoder_hdims,  nn.LeakyReLU, initializer='normal')
        self.encoder = nn.Sequential(*encoder_layers)
        self.z_mu    = nn.Linear(encoder_hdims[-1], zdim)
        self.z_var   = nn.Linear(encoder_hdims[-1], zdim)

    def forward(self,x,c):
        z = self.encode(x=x,c=c)
        pi, mu, sigma, x_recon = self.decode(c=c,z=z)
        return pi, mu, sigma, x_recon 
    
    def encode(self, x,c):
        xc = torch.cat([x,c], dim=1)
        z  = self.encoder(xc)
        z_mu = self.z_mu(z)
        z_var = self.z_var(z)
        z_var = torch.exp(z_var)
        eps_sample = torch.randn(
            size=z_mu.shape,dtype=torch.float32).to(self.device)
        z_sample   = z_mu + torch.sqrt(z_var+1e-10)*eps_sample
        # Vmf
        if self.sn:
            z_sample   = F.normalize(z_sample, p=2,  dim = -1)
        return z_sample 

    def decode(self, c, z):
        cz = torch.cat([c,z], dim=1)
        pi = self.mdn_pi(cz)
        mu = self.mdn_mu(cz)
        sigma = self.mdn_sigma(cz)
        recon_x_distribution = torch.distributions.Normal(loc=mu, scale=sigma)
        pi_distribution = torch.distributions.OneHotCategorical(logits=pi)
        pi = pi_distribution.sample()
        x_recon =recon_x_distribution.rsample()
        return pi, mu, sigma, x_recon 
    
    def gaussian_probability(self,sigma, mu, target):
        ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
        print('target', target)
        target = target.expand_as(sigma)
        print('target', target)
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
        return ret


    def mdn_loss_fn(self, pi, sigma, mu, target):
        print('pi', pi)
        prob = pi * self.gaussian_probability(sigma, mu, target)
        print('prob', prob)
        # prob = torch.tensor([0,0,0,0,1])
        nll = -torch.log(torch.sum(prob, dim=1))
        print('nll', nll)
        return torch.mean(nll)

    def update(self, x, c, q):
        pi, sigma, mu, x_recon = self.forward(x, c)
        recon_loss = self.mdn_loss_fn(pi, sigma, mu, x_recon)
        print('recon_loss', recon_loss)

class SMORE:
    def __init__(self, xdim=20, cdim=3, zdim=5, hdims=[128, 128], n_components=5, lr=1e-3, device="cpu"):
        self.xdim         = xdim 
        self.cdim         = cdim 
        self.zdim         = zdim 
        self.hdims        = hdims
        self.n_components = n_components
        self.device       = device 
        # self.encoder      = 
        self.mlpg         = MLPG(xdim=self.xdim, cdim=self.cdim, zdim=self.zdim, hdims=self.hdims, device=self.device).to(self.device)
        self.optimizer    = torch.optim.Adam(params=self.mlpg.parameters(), lr=lr)


if __name__=="__main__":
    model = MLPG(spectral_norm=False)
    x = torch.randn(size=(1, 2))
    c = torch.randn(size=(1, 6))
    q = torch.randn(size=(1,1))
    # z = torch.randn(size=(1, 5)).to('cpu')
    # z = F.normalize(z, p=2, dim=1)
    z = model.encode(x=x, c=c)
    model.decode(c=c, z=z)
    pi, sigma, mu, x_recon = model.forward(x, c)
    recon_loss = model.mdn_loss_fn(pi, sigma, mu, x_recon)
    print('recon_loss', recon_loss)