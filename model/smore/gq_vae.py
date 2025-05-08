import numpy as np
import random
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys 
sys.path.append('..')
from model.utils.utils import torch2np, np2torch, get_runname


class GumbelQuantizer(nn.Module):
    def __init__(
                self, 
                z_dim, 
                embedding_num, 
                embedding_dim,
                tau_scale = 1.0,
                kld_scale = 5e-4,
                straight_through=False,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                ):
        super(GumbelQuantizer, self).__init__()
        self.embedding_num    = embedding_num
        self.embedding_dim    = embedding_dim
        self.straight_through = straight_through
        self.device     = device
        self.tau_scale  = tau_scale
        self.kld_scale  = kld_scale
        self.projection = nn.Linear(z_dim, self.embedding_num)
        self.embedding  = nn.Embedding(self.embedding_num, embedding_dim)
        # self.init_embedding()

    def init_embedding(self):
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data.fill_diagonal_(1)

    def zero_grad_for_non_diagnoal_weights(self):
        for row, grads in enumerate(self.embedding.weight.grad):
            grads[:] = 0

    def forward(self, z, q, hard=False):
        # force hard = True when we are in eval mode, as we must quantize
        hard   = self.straight_through if self.training else True
        logits = self.projection(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.tau_scale, dim=1, hard=hard)
        z_q = torch.matmul(soft_one_hot, self.embedding.weight).to(self.device)
        # + kl divergence to the prior loss
        qy   = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy + 1e-10), dim=1).mean()
        # 
        diff  = diff
        return z_q, diff
    
class GumbelQuantizedVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        name     = 'GQVAE',              
        x_dim    = 784,              # input dimension
        c_dim    = 10,               # condition dimension
        z_dim    = 15,               # latent dimension
        h_dims   = [64,32],          # hidden dimensions of encoder (and decoder)
        embedding_num   = 10,        # For VQ parameters
        embedding_dim   = 3,         # For VQ parameters
        tau_scale = 1.0,             # For VQ parameters
        kld_scale = 5e-4,            # For VQ parameters 
        actv_enc = nn.ReLU(),        # encoder activation
        actv_dec = nn.ReLU(),        # decoder activation
        actv_q   = nn.Softplus(),    # q activation
        actv_out = None,             # output activation
        device   = 'cpu'
        ):
        """
            Initialize
        """
        super(GumbelQuantizedVariationalAutoEncoder, self).__init__()
        self.name   = name
        self.x_dim  = x_dim
        self.c_dim  = c_dim
        self.z_dim  = z_dim
        self.h_dims = h_dims
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.tau_scale = tau_scale
        self.kld_scale = kld_scale
        self.actv_enc  = actv_enc
        self.actv_dec  = actv_dec
        self.actv_q    = actv_q
        self.actv_out  = actv_out
        self.device    = device
        self.histogram = torch.zeros(self.embedding_dim).to(self.device)
        # Initialize VQ class
        self.GQ = GumbelQuantizer(self.z_dim, self.embedding_num, self.embedding_dim, self.tau_scale, self.kld_scale).to(self.device)
        # Initialize layers
        self.init_layers()
        self.init_params()
                
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        
        # Encoder part
        h_dim_prev = self.x_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims):
            self.layers['enc_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['enc_%02d_actv'%(h_idx)] = \
                self.actv_enc
            h_dim_prev = h_dim
        self.layers['z_lin']  = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        
        # Decoder part
        h_dim_prev = self.z_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims[::-1]):
            self.layers['dec_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            if self.actv_dec is not None:
                self.layers['dec_%02d_actv'%(h_idx)] = \
                    self.actv_dec
            h_dim_prev = h_dim
        self.layers['out_lin'] = nn.Linear(h_dim_prev,self.x_dim,bias=True)
        
        # Append parameters
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.gqvae_parameters = nn.ParameterDict(self.param_dict)
        
    def xc_to_z(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x to z
        """
        if c is not None:
            net = torch.cat((x,c), dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            net = self.layers['enc_%02d_actv'%(h_idx)](net)
        z = self.layers['z_lin'](net)
        return z
    
    def zc_to_x_recon(
        self,
        z = torch.randn(2,16),
        c = torch.randn(2,10),
        q = torch.randn(2,1)
        ):
        """
            z and c to x_recon
        """
        net, _ = self.GQ(z, q, hard=True)
        count = torch.sum(net, axis=0)
        self.histogram += count.detach()
        if c is not None:
            net = torch.cat((net,c),dim=1)
        else:
            pass
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            if self.actv_dec is not None:
                net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon

    def z_q_to_x_recon(
        self,
        z_q,
        c
        ):
        """
            z and c to x_recon
        """
        net = torch.cat((z_q,c),dim=1)
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon

    def xc_to_x_recon(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10),
        q = torch.randn(2,1)
        ):
        """
            x to x_recon
        """
        z = self.xc_to_z(x=x, c=c)
        x_recon = self.zc_to_x_recon(z=z, q=q, c=c)
        return x_recon

    def sample_x(
        self,
        c = torch.randn(2,10),
        n_sample = 1
        ):
        """
            sample x from codebook
        """
        random_integers  = np.random.permutation(self.embedding_num)[:n_sample]
        random_embedding = self.GQ.embedding.weight.data[random_integers, :]
        x_sample = self.z_q_to_x_recon(z_q=random_embedding, c=c).detach().cpu().numpy()
        return x_sample

    def init_params(self,seed=0):
        """
            Initialize parameters
        """
        # Fix random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Init
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def loss_recon(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q          = torch.randn(2,1),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0
        ):
        """
            Recon loss
        """
        x_recon = self.xc_to_x_recon(x=x, c=c)
        if (LOSS_TYPE == 'L1') or (LOSS_TYPE == 'MAE'):
            errs = torch.mean(torch.abs(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L2') or (LOSS_TYPE == 'MSE'):
            errs = torch.mean(torch.square(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L1+L2') or (LOSS_TYPE == 'EN'):
            errs = torch.mean(
                0.5*(torch.abs(x-x_recon)+torch.square(x-x_recon)),axis=1)
        else:
            raise Exception("VAE:[%s] Unknown loss_type:[%s]"%
                            (self.name,LOSS_TYPE))
        return recon_loss_gain*torch.mean(errs*q)
        # return recon_loss_gain*torch.mean(errs)

    def loss_total(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q          = torch.randn(2,1),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0
        ):
        """
            Total loss
        """
        loss_recon_out = self.loss_recon(
            x               = x,
            c               = c,
            q               = q,
            LOSS_TYPE       = LOSS_TYPE,
            recon_loss_gain = recon_loss_gain
        )
        z = self.xc_to_z(x, c)
        _, loss_vq = self.GQ(z, q, hard=True)
        loss_total_out = loss_recon_out + loss_vq
        info           = {'loss_total_out' : loss_total_out,
                          'loss_recon_out' : loss_recon_out,
                          'loss_vq'        : loss_vq}
        return loss_total_out, info

    def debug_plot_img(
        self,
        x_train_np     = np.zeros((60000,784)),  # to plot encoded latent space 
        y_train_np     = np.zeros((60000)),      # to plot encoded latent space 
        c_train_np     = np.zeros((60000,10)),   # to plot encoded latent space
        x_test_np      = np.zeros((10000,784)),
        c_test_np      = np.zeros((10000,10)),
        c_vecs         = np.eye(10,10),
        n_sample       = 10,
        img_shape      = (28,28),
        img_cmap       = 'gray',
        figsize_image  = (10,3.25),
        figsize_latent = (10,3.25),
        DPP_GEN        = False,
        dpp_hyp        = {'g':1.0,'l':0.1}
        ):
        """
            Debug plot
        """
        n_train       = x_train_np.shape[0]
        x_train_torch = np2torch(x_train_np, device=self.device)
        # Reconstruct
        x_test_torch  = np2torch(x_test_np, device=self.device)
        c_test_torch  = np2torch(c_test_np, device=self.device)
        n_test        = x_test_np.shape[0]
        rand_idxs     = np.random.permutation(n_test)[:n_sample]
        x_recon = self.xc_to_x_recon(x=x_test_torch[rand_idxs,:], c=c_test_torch[rand_idxs]).detach().cpu().numpy()
        # Generation
        random_integers  = np.random.permutation(self.embedding_num)[:n_sample]
        random_embedding = self.GQ.embedding.weight.data[random_integers, :]
        x_sample = self.z_q_to_x_recon(z_q=random_embedding, c=c_test_torch[random_integers]).detach().cpu().numpy()
        # Plot images to reconstruct
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_test_np[rand_idxs[s_idx],:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Images to Reconstruct",fontsize=15);plt.show()
        
        # Plot reconstructed images
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_recon[s_idx,:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Reconstructed Images",fontsize=15);plt.show()

        # Plot generated images
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_sample[s_idx,:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Generated Images",fontsize=15);plt.show()