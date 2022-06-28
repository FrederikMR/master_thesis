# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:18:47 2021

@author: Frederik
"""

#%% Sources

"""
Sources used:
https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
https://github.com/sarthak268/Deep_Neural_Networks/blob/master/Autoencoder/Variational_Autoencoder/generative_vae.py
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb
"""


#%% Modules

import torch
from torch import nn
from typing import List, Any
        
#%% Using direct results for normal 
#assumptions with variance

#The training script should be modified for the version below.
class AE_3d(nn.Module):
    def __init__(self,
                 fc_h: List[int] = [3, 100, 2],
                 fc_g: List[int] = [2, 100, 3],
                 fc_h_act: List[Any] = [nn.ELU, nn.Identity],
                 fc_g_act: List[Any] = [nn.ELU, nn.Identity],
                 fc_mu_act: List[Any] = [nn.Identity],
                 fc_var_act: List[Any] = [nn.Sigmoid]
                 ):
        super(AE_3d, self).__init__()
    
        self.fc_h = fc_h
        self.fc_g = fc_g
        self.fc_h_act = fc_h_act
        self.fc_g_act = fc_g_act
        
        self.num_fc_h = len(fc_h)
        self.num_fc_g = len(fc_g)
        
        self.encoder = self.encode()
        self.decoder = self.decode()
        self.loss = nn.MSELoss()
        
    def encode(self):
        
        layer = []
        
        for i in range(1, self.num_fc_h):
            layer.append(nn.Linear(self.fc_h[i-1], self.fc_h[i]))
            layer.append(self.fc_h_act[i-1]())
            #input_layer.append(self.activations_h[i](inplace=True))
            
        return nn.Sequential(*layer)
        
    def decode(self):
        
        layer = []
        
        for i in range(1, self.num_fc_g):
            layer.append(nn.Linear(self.fc_g[i-1], self.fc_g[i]))
            layer.append(self.fc_g_act[i-1]())
            
        return nn.Sequential(*layer)
    
    def forward(self, x):
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        mse = self.loss(x, x_hat)
        
        return z, x_hat, mse
    
    def h(self, x):
        
        z = self.encoder(x)
        
        return z
        
    def f(self, z):
        
        x_hat = self.decoder(z)
        
        return x_hat
