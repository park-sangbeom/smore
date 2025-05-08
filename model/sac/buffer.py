

import numpy as np 
import torch
import collections
import random

class ReplayBuffer():
    def __init__(self, buffer_limit, device="cpu"):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device 
    def put(self,item):
        self.buffer.append(item)

    def sample(self,n):
        mini_batch = random.sample(self.buffer,n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for item in mini_batch:
            s, a, r, s_prime, done = item
            s_list.append(s)
            a_list.append(a)
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_list.append([done_mask])
        s_list = torch.tensor(np.array(s_list), dtype = torch.float).to(self.device)
        a_list = torch.tensor(np.array(a_list), dtype = torch.float).to(self.device)
        r_list = torch.tensor(np.array(r_list), dtype = torch.float).to(self.device)
        s_prime_list = torch.tensor(np.array(s_prime_list), dtype = torch.float).to(self.device)
        done_mask_list = torch.tensor(np.array(done_mask_list), dtype = torch.float).to(self.device)
        return s_list, a_list, r_list, s_prime_list, done_mask_list

    def size(self):
        return len(self.buffer)