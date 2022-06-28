# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:44:58 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
http://adamlineberry.ai/vae-series/vae-code-experiments
https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
"""

#%% Modules

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.realpath(currentdir))
parentdir = os.path.dirname(os.path.realpath(parentdir))
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch import nn
import math
import argparse
pi = math.pi

#Own files
from rm_dgm import rm_generative_models
from VAE_surface3d import VAE_3d

#%% Fun

def circle_fun(theta, mu = np.array([1.,1.,1.]), r=1):
    
    x1 = r*torch.cos(theta)+mu[0]
    x2 = r*torch.sin(theta)+mu[1]
    x3 = torch.zeros(1)+mu[2]
    
    return x1, x2, x3

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='sim_data/circle.csv', 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/circle', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--MAX_ITER', default=100,
                        type=int)
    parser.add_argument('--eps', default=0.1,
                        type=int)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--alpha', default=1,
                        type=float)
    parser.add_argument('--lr', default=0.0001,
                        type=float)

    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/VAE_circle_epoch_100000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()

    #Hyper-parameters
    latent_dim = 2
    
    #Loading files
    
    #Loading data
    df = pd.read_csv(args.data_path, index_col=0)
    DATA = torch.Tensor(df.values)
    DATA = torch.transpose(DATA, 0, 1)
    
    #Loading model
    model = VAE_3d(fc_h = [3, 100],
                     fc_g = [latent_dim, 100, 3],
                     fc_mu = [100, latent_dim],
                     fc_var = [100, latent_dim],
                     fc_h_act = [nn.ELU],
                     fc_g_act = [nn.ELU, nn.Identity],
                     fc_mu_act = [nn.Identity],
                     fc_var_act = [nn.Sigmoid]).to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Latent coordinates
    zx = torch.tensor(pi/2).view(1).to(args.device)
    zy = torch.tensor(0.).view(1).to(args.device)
    
    #Coordinates on the manifold
    y = (torch.tensor(circle_fun(zy))).float()
    
    #Mean of approximate posterier
    hy = model.h(y)
    gy = model.g(hy)
    
    #Loading module
    rm = rm_generative_models(model.h, model.g, args.device)
    
    x = (torch.tensor(circle_fun(zx))).float()
    hx = model.h(x)
    gx = model.g(hx)
    
    gamma_linear = rm.linear_inter(hx,hy,args.T)
    g_linear = model.g(gamma_linear)
    loss, z_geodesic = rm.compute_geodesic(gamma_linear, 100000)
    g_geodesic = model.g(z_geodesic)
    L_geodesic = rm.arc_length(g_geodesic)
    L_linear = rm.arc_length(g_linear)
    
    checkpoint = args.save_path+'.pt'
    torch.save({'loss': loss,
                'G_old': g_linear,
                'G_new': g_geodesic,
                'L_old': L_linear.item(),
                'L_new': L_geodesic.item(),
                'z_linear': gamma_linear,
                'z_geodesic': z_geodesic,
                'gx': gx,
                'gy': gy}, 
               checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    