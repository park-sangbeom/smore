

import sys 
sys.path.append("..")
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ActorClass(nn.Module):
    def __init__(self, action_dim=2,
                       state_dim=17,
                       lr=3e-4, 
                       init_alpha=0.1,
                       lr_alpha=3e-4):
        super(ActorClass,self).__init__()
        # Gaussian Distribution
        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc_mean = nn.Linear(256,action_dim)
        self.fc_std = nn.Linear(256,action_dim)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.target_entropy = -action_dim
        # Autotuning Alpha
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha],lr = lr_alpha)  

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        # std : softplus or ReLU activate function [Because It should be higher than 0]
        std = F.softplus(self.fc_std(x))
        Gaussian = Normal(mean,std)
        action = Gaussian.rsample()
        log_prob = Gaussian.log_prob(action)
        # action range : -1 ~ 1
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_p(self,q1,q2,mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = (-min_q - entropy)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class CriticClass(nn.Module):
    def __init__(self, state_dim=17,
                       action_dim=2, 
                       lr=1e-4,
                       tau=0.005):
        super(CriticClass,self).__init__()
        self.fc_s = nn.Linear(state_dim,128)
        self.fc_a = nn.Linear(action_dim,128)
        self.fc_cat = nn.Linear(256,256)
        self.fc_out = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.tau = tau 

    def forward(self,x,a):
        x = F.relu(self.fc_s(x))
        a = F.relu(self.fc_a(a))
        cat = torch.cat([x,a], dim=1)
        q = F.relu(self.fc_cat(cat))
        q_value = self.fc_out(q)

        return q_value

    def train_q(self,target,mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s,a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    # DDPG soft_update
    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

def get_target(pi, q1, q2, mini_batch, gamma=0.99):
    s, a, r, s_prime, done = mini_batch
    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy.mean())
    return target 