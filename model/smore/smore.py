import sys
from pathlib import Path
BASEDIR = str(Path(__file__).parent)
sys.path.append(BASEDIR)
sys.path.append("..")
import numpy as np 
import random as rd
import math
import torch 
import torch.nn as nn
from scipy.spatial import distance
from model.utils.gaussian_path_planner import GaussianRandomPathClass, kernel_levse, kernel_se, soft_squash
from model.utils.utils import np2torch, get_linear_layer
import torch.nn.functional as F

class Encoder(nn.Module): 
    def __init__(self, xdim=20, cdim=3, zdim=10, hdims=[128,128], device='cpu', sn=True):
        super(Encoder, self).__init__()

        hidden_actv=nn.Tanh; loc_actv=nn.Tanh; scale_actv=nn.Sigmoid
        hdims.insert(0, xdim+cdim)
        layers = get_linear_layer(hdims, hidden_actv)
        self.layers = nn.Sequential(*layers)
        # self.z_loc       = nn.Linear(hdims[-1], zdim)
        # self.z_log_scale = nn.Linear(hdims[-1], zdim)

        nn.Sequential(*get_linear_layer(hdim=[hdims[-1], zdim], hidden_actv=loc_actv))
        self.z_log_scale = nn.Sequential(*get_linear_layer(hdim=[hdims[-1], zdim], hidden_actv=scale_actv))
        self.z_loc       = nn.Sequential(*get_linear_layer(hdim=[hdims[-1], zdim], hidden_actv=loc_actv))
        self.device =device
        self.sn = sn 
    
    def xc_to_z(self,x,c):
        xc = torch.cat([x,c], dim=-1)
        xc = self.layers(xc)
        z_mu = self.z_loc(xc)
        z_var = self.z_log_scale(xc)
        z_var = torch.exp(z_var)
        eps_sample = torch.randn(
            size=z_mu.shape,dtype=torch.float32).to(self.device)
        z_sample   = z_mu + torch.sqrt(z_var+1e-10)*eps_sample
        # Vmf
        if self.sn:
            z_sample   = F.normalize(z_sample, p=2,  dim = -1)
        return z_sample, z_mu, z_var
    
    def forward(self, x, c):
        xc = torch.cat([x,c], dim=-1)
        xc = self.layers(xc)
        z_loc       = self.z_loc(xc)
        # z_loc       = torch.nan_to_num_(z_loc)
        z_log_scale = self.z_log_scale(xc)
        # z_log_scale = torch.nan_to_num_(z_log_scale)
        z_distribution = torch.distributions.Normal(loc=z_loc, scale=z_log_scale)
        return z_distribution
        # z_mu = self.z_loc(xc)
        # z_var = self.z_log_scale(xc)
        # z_var = torch.exp(z_var)
        # eps_sample = torch.randn(
        #     size=z_mu.shape,dtype=torch.float32).to(self.device)
        # z_sample   = z_mu + torch.sqrt(z_var+1e-10)*eps_sample
        # # Vmf
        # if self.sn:
        #     z_sample   = F.normalize(z_sample, p=2,  dim = -1)
        # return z_sample 
    
class CategoricalNetwrok(nn.Module):
    def __init__(self, xdim=20, cdim=3, zdim=5, hdims=[128,128], n_components=5):
        super().__init__()
        self.n_components = n_components
        self.xdim = xdim 
        hdims.append(zdim+cdim)

        layers = get_linear_layer(list(reversed(hdims)), hidden_actv=nn.Tanh)
        self.layers = nn.Sequential(*layers)
        self.MixtureLogit = nn.Sequential(*get_linear_layer(hdim=[hdims[0], n_components], hidden_actv=nn.Softmax))
    
    def forward(self, z_sample, c):
        z_c_concat = torch.cat((z_sample, c), axis=1)
        out = self.layers(z_c_concat)
        logits = self.MixtureLogit(out)
        return torch.distributions.OneHotCategorical(logits=logits)

class MixtureComponentNetwork(nn.Module):
    def __init__(self, xdim=20, cdim=3, zdim=5, hdims=[128,128], n_components=5, device='cpu'):
        super().__init__()
        self.n_components = n_components
        self.xdim = xdim 
        hdims.append(zdim+cdim)
        self.layers = nn.Sequential(*get_linear_layer(list(reversed(hdims)), hidden_actv=nn.Tanh))
        
        self.recon_loc       = nn.Sequential(*get_linear_layer(hdim=[hdims[0], n_components * xdim], hidden_actv=nn.Tanh))
        self.recon_log_scale = nn.Sequential(*get_linear_layer(hdim=[hdims[0], n_components * xdim], hidden_actv=nn.Sigmoid))
        self.device = device

    def forward(self, z_sample, c):
        z_c_concat = torch.cat((z_sample, c), axis=1)
        out = self.layers(z_c_concat)        
        x_loc = self.recon_loc(out).reshape(-1, self.n_components, self.xdim)
        # original_shape = (out.shape[0], self.n_components, self.xdim)
        # scaling_factors = torch.zeros(original_shape)
        # for i in range(self.xdim):
        #     if i % 3 == 0:
        #         scaling_factors[:, :, i] = 1.0
        #     elif i % 3 == 1:
        #         scaling_factors[:, :, i] = 0.35
        #     else:  # i % 3 == 2
        #         scaling_factors[:, :, i] = 1.1
        # x_loc = x_loc * scaling_factors
        x_scale = self.recon_log_scale(out).reshape(-1, self.n_components, self.xdim)
        recon_x_distribution = torch.distributions.Normal(x_loc,x_scale)
        return recon_x_distribution
    

class Decoder(nn.Module):
    def __init__(self, xdim=20, cdim=3, zdim=10, hdims=[128,128], n_components=5, device='cpu'):
        super(Decoder, self).__init__()
        self.pi_network     = CategoricalNetwrok(xdim=xdim, cdim=cdim, zdim=zdim, hdims=hdims, n_components=n_components)
        self.normal_network = MixtureComponentNetwork(xdim=xdim, cdim=cdim, zdim=zdim, hdims=hdims, n_components=n_components, device=device)
    
    def forward(self, z_sample, c):
        pi = self.pi_network(z_sample, c)
        mean = self.normal_network(z_sample, c)
        return pi, mean 

    def sample(self, z_sample, c):
        pi, normal = self.forward(z_sample, c)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.loc, dim=1)
        return samples
    

class SMORE(nn.Module): 
    def __init__(self, xdim=20, cdim=3, zdim=5, hdims=[128,128], n_components=5, lr=1e-3, device='cpu'):
        super(SMORE, self).__init__()
        self.xdim=xdim
        self.cdim=cdim 
        self.zdim=zdim 
        self.hdims=hdims
        self.n_components=n_components
        self.encoder = Encoder(xdim=xdim, cdim=cdim, zdim=zdim, hdims=hdims, device=device)
        self.decoder = Decoder(xdim=xdim, cdim=cdim, zdim=zdim, hdims=hdims, n_components=n_components, device=device)
        self.prior_distribution = torch.distributions.Normal(
        loc = torch.zeros(1,zdim).to(device),
        scale = torch.ones(1,zdim).to(device))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.grp    = GaussianRandomPathClass(name='GRP',kernel=kernel_levse)
        self.device = device

    def encode(self, x, c):
        z_distribution = self.encoder(x, c)
        return z_distribution 
    
    def decode(self, z_distribution, c):
        # z_sample = z_distribution
        z_sample = z_distribution.rsample()
        mixture_mode, mixture_components = self.decoder(z_sample, c)
        return mixture_mode, mixture_components 
    
    def forward(self, x, c):
        z_distribution = self.encode(x,c)
        mixture_mode, mixture_components = self.decode(z_distribution, c)
        return z_distribution, mixture_mode, mixture_components 

    def update(self, batch):
        x_batch, c_batch, reward_batch = batch["x"], batch["c"], batch["reward"]
        z_distribution, mixture_mode, mixture_components = self.forward(x_batch, c_batch)
        # Compute the loss
        unscaled_kld_loss   = torch.sum(torch.distributions.kl_divergence(z_distribution, self.prior_distribution), dim=1)
        loglik = mixture_components.log_prob(x_batch.unsqueeze(1).expand_as(mixture_components.loc))
        loglik = torch.sum(loglik, dim=2)
        unscaled_recon_loss = -torch.logsumexp(mixture_mode.logits + loglik, dim=1)
        
        kld_loss            = torch.mean(reward_batch*unscaled_kld_loss,  dim=0)
        recon_loss          = torch.mean(reward_batch*unscaled_recon_loss,dim=0)
        
        total_loss          = kld_loss + recon_loss
        mean_unscaled_kld_loss   = torch.mean(unscaled_kld_loss,  dim=0)
        mean_unscaled_recon_loss = torch.mean(unscaled_recon_loss,dim=0)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)

        self.optimizer.step()
        return mean_unscaled_kld_loss, mean_unscaled_recon_loss, kld_loss, recon_loss, total_loss

    """ Prior: Multi trajectories """
    def random_explore(self, n_sample=50):
        t_test = np.linspace(start=0.65,stop=1.05,num=10).reshape((-1,1))
        t_anchor = np.array([0.65,1.05]);x_anchor = np.array([0,0])
        self.grp.set_data(t_anchor= t_anchor.reshape((-1,1)),
                    x_anchor = np.array([x_anchor]).T,
                    l_anchor = np.array([[1,0.1]]).T,
                    t_test   = t_test,
                    l_test   = np.ones((len(t_test),1)),
                    hyp_mean = {'g':0.1,'l':0.08,'w':1e-6},
                    hyp_var  = {'g':0.05,'l':0.1,'w':1e-6})
        ys, xs = self.grp.sample(n_sample=n_sample)
        return xs, ys 

    """ Posterior: Single trajectory """
    @torch.no_grad()
    def exploit(self, c):
        t_test = np.linspace(start=0.65,stop=1.05,num=10).reshape((-1,1))
        z_sample = self.prior_distribution.sample().to(self.device)
        recon_x = self.decoder.sample(z_sample, c)
        x_anchor_recon = recon_x.reshape(-1).numpy()
        t_anchor = np.linspace(start=0.65,stop=1.05,num=10).reshape((-1,1))
        self.grp.set_data(t_anchor    = t_anchor, 
                          x_anchor    = np.array([x_anchor_recon]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = t_test,
                          l_test      = np.ones((len(t_test),1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
        ys, xs = self.grp.mean_traj(SQUASH=True)
        return xs, ys, x_anchor_recon

    def random_explore_3d(self, n_sample=50, scenario=1):
        # if scenario==1:
        #     rand_x = np.random.uniform(low=0.6, high=0.83)
        #     rand_y = np.random.uniform(low=-0.35, high=0.)
        # elif scenario==2:
        #     rand_x = np.random.uniform(low=0.6, high=0.7)
        #     rand_y = np.random.uniform(low=-0.35, high=0.35)
        # elif scenario==3:
        #     if rd.random()<0.5:
        #         rand_x = np.random.uniform(low=0.73, high=0.83)
        #         rand_y = np.random.uniform(low=-0.35, high=-0.1)
        #     else: 
        #         rand_x = np.random.uniform(low=0.6, high=0.7)
        #         rand_y = np.random.uniform(low=0.1, high=0.35)
        # elif scenario==4:
        #     rand_x = np.random.uniform(low=0.6, high=0.83)
        #     rand_y = np.random.uniform(low=0, high=0.35)
        # elif scenario==5:
        #     rand_x = np.random.uniform(low=0.73, high=0.83)
        #     rand_y = np.random.uniform(low=-0.35, high=0.35)
        # elif scenario==6:
        #     if rd.random()<0.5:
        #         rand_x = np.random.uniform(low=0.6, high=0.7)
        #         rand_y = np.random.uniform(low=-0.35, high=-0.1)
        #     else: 
        #         rand_x = np.random.uniform(low=0.7, high=0.8)
        #         rand_y = np.random.uniform(low=0.1, high=0.35)
        rand_x = np.random.uniform(low=0.6, high=0.83)
        rand_y = np.random.uniform(low=-0.35, high=0.35)
        t_anchor = np.linspace(start=0.6,stop=rand_x,num=2).reshape((-1,1))
        t_test = np.linspace(start=0.6,stop=rand_x,num=10).reshape((-1,1))
        self.grp.set_data(t_anchor    = t_anchor,
           x_anchor    = np.array([[0., rand_y]]).T,
           t_test      = t_test,
           l_anchor = np.array([[1,0.6]]).T,
           l_test   = np.ones((10,1)),
           hyp_mean    = {'g':0.1,'l':0.1,'w':1e-6},
           hyp_var     = {'g':0.1,'l':0.2,'w':1e-6},
        #    hyp_mean = {'g':0.1,'l':0.1,'w':1e-6},
        #     hyp_var  = {'g':0.04,'l':0.06,'w':1e-6},
           APPLY_EPSRU = False)
        y_trajs, x_traj = self.grp.sample(n_sample=n_sample)
        y_trajs = soft_squash(y_trajs,x_min=-0.35,x_max=0.35, margin=0.05)
        z_traj = np.array([[1.1], [1.1], [1.07], [1.06], [1.03], [1.0], [0.99], [0.98], [0.96], [0.92]])
        x_trajs = [x_traj[:] for _ in range(n_sample)]
        z_trajs = [z_traj[:] for _ in range(n_sample)]

        # self.grp.set_data(t_anchor = np.linspace(start=0.0,stop=1,num=2).reshape((-1,1)),
        #    x_anchor    = np.array([[0.6, rand_x]]).T,
        #    t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
        #    l_anchor = np.array([[1,1]]).T,
        #    l_test   = np.ones((10,1)),
        #    hyp_mean    = {'g':0.5,'l':0.5,'w':1e-6},
        #    hyp_var     = {'g':0.1,'l':0.5,'w':1e-6},
        #    APPLY_EPSRU = False)
        # x_trajs, _ = self.grp.sample(n_sample=n_sample)
        # x_trajs = soft_squash(x_trajs,x_min=0.6,x_max=0.8, margin=0.05)

        # self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=2).reshape((-1,1)),
        #    x_anchor    = np.array([[0., 0.]]).T,
        #    t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
        #    l_anchor = np.array([[1,0.1]]).T,
        #    l_test   = np.ones((10,1)),
        #    hyp_mean    = {'g':0.1,'l':0.1,'w':1e-6},
        #    hyp_var     = {'g':0.1,'l':0.2,'w':1e-6},
        #    APPLY_EPSRU = False)
        # y_trajs, _ = self.grp.sample(n_sample=n_sample)
        # y_trajs = soft_squash(y_trajs,x_min=-0.33,x_max=0.33, margin=0.05)

        # self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=2).reshape((-1,1)),
        #    x_anchor    = np.array([[1.2, 1.0]]).T,
        #    t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
        #    l_anchor = np.array([[1,1]]).T,
        #    l_test   = np.ones((10,1)),
        #    hyp_mean    = {'g':0.5,'l':0.5,'w':1e-6},
        #    hyp_var     = {'g':0.1,'l':0.5,'w':1e-6},
        #    APPLY_EPSRU = False)
        # z_trajs, _ = self.grp.sample(n_sample=n_sample)
        # z_trajs = soft_squash(z_trajs,x_min=0.9,x_max=1.2, margin=0.05)
        return x_trajs,y_trajs,z_trajs

    @torch.no_grad()
    def exploit_3d(self, c=torch.randn(64,4)):        
        # Get reconstructed anchors
        z_sample = self.prior_distribution.sample().to(self.device)
        x_anchor_recon = self.decoder.sample(z_sample, c) 
        print(x_anchor_recon)       
        x_anchor_recon = x_anchor_recon.reshape(10,-1).T.numpy()  
        self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
                          x_anchor    = np.array([x_anchor_recon[1,:]]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
                          l_test      = np.ones((10,1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
        xs, _ = self.grp.mean_traj(x_min=0.6,x_max=0.83, margin=0.01)

        self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
                          x_anchor    = np.array([x_anchor_recon[0,:]]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
                          l_test      = np.ones((10,1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
        ys, _ = self.grp.mean_traj(x_min=-0.35,x_max=0.35, margin=0.01)
        zs = np.array([[1.1], [1.1], [1.07], [1.06], [1.03], [1.0], [0.99], [0.98], [0.96], [0.92]])
        return xs,ys,zs,x_anchor_recon

    # @torch.no_grad()
    # def exploit_3d(self, c=torch.randn(64,4)):        
    #     # Get reconstructed anchors
    #     z_sample = self.prior_distribution.sample().to(self.device)
    #     x_anchor_recon = self.decoder.sample(z_sample, c)        
    #     x_anchor_recon = x_anchor_recon.reshape(10,-1).T.numpy()  
    #     self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
    #                       x_anchor    = np.array([x_anchor_recon[0,:]]).T,
    #                       l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
    #                       t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
    #                       l_test      = np.ones((10,1)),
    #                       hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
    #                       hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
    #     xs, _ = self.grp.mean_traj(x_min=0.6,x_max=0.83, margin=0.01)

    #     self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
    #                       x_anchor    = np.array([x_anchor_recon[1,:]]).T,
    #                       l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
    #                       t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
    #                       l_test      = np.ones((10,1)),
    #                       hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
    #                       hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
    #     ys, _ = self.grp.mean_traj(x_min=-0.34,x_max=0.34, margin=0.01)

    #     self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
    #                       x_anchor    = np.array([x_anchor_recon[2,:]]).T,
    #                       l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
    #                       t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
    #                       l_test      = np.ones((10,1)),
    #                       hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
    #                       hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
    #     zs, _ = self.grp.mean_traj(x_min=0.9,x_max=1.1, margin=0.01)
    #     return xs,ys,zs,x_anchor_recon

    