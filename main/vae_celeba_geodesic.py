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

#Modules
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse

import numpy as np

from scipy import ndimage

#Own files
#from rm_computations import rm_data
from VAE_celeba import VAE_CELEBA
from rm_dgm import rm_generative_models

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default="../../Data/CelebA/celeba", 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/celeba_geodesic.pt', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=100000,
                        type=int)
    parser.add_argument('--T', default=99,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)
    parser.add_argument('--size', default=64,
                        type=float)

    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/celeba/celeba_epoch_50000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    img_size = 32
    transform=transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_data = dset.ImageFolder('celeba/',
                                      transform = transform)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False)

    images, _ = next(iter(dataloader))
    
    img_base = np.transpose(np.asarray(images[0]), axes=(1,2,0))

    num_rotate = 100
    theta = np.linspace(0,2*np.pi,num_rotate)

    theta_degrees = theta*180/np.pi

    rot = []
    for v in theta_degrees:
        rot.append(ndimage.rotate(img_base, v, reshape=False))
    rot = np.stack(rot)
    
    rot = np.einsum('ijkm->imjk', rot)
    
    DATA = torch.Tensor(rot).to(args.device)
    
    #Plotting the trained model
    model = VAE_CELEBA().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    rm = rm_generative_models(model.h, model.g, args.device)
    
    theta0 = np.pi/2
    idx = np.argmin(np.abs(theta-theta0))
    img0 = rot[idx].reshape(1,3,32,32)
    
    thetaT = 0.0
    idx = np.argmin(np.abs(theta-thetaT))
    imgT = rot[idx].reshape(1,3,32,32)
    
    img0 = torch.Tensor(img0).to(args.device)
    imgT = torch.Tensor(imgT).to(args.device)
    
    hx = model.h(img0).to(args.device)
    hy = model.h(imgT).to(args.device)
            
    gamma_linear = rm.linear_inter(hx, hy, args.T)
    
    loss, gammaz_geodesic = rm.compute_geodesic(gamma_linear,epochs=args.epochs)
    gamma_g_geodesic = model.g(gammaz_geodesic)
    gamma_g_linear = model.g(gamma_linear)
    
    gammaz_geodesic = gammaz_geodesic.detach()
    gamma_g_linear = gamma_g_linear.detach()
    
    L_linear = rm.arc_length(gamma_g_linear)
    L_geodesic = rm.arc_length(gamma_g_geodesic)
        
    G_plot = torch.cat((gamma_g_linear.detach(), gamma_g_geodesic.detach()), dim = 0)
    
    arc_length = ['{0:.4f}'.format(L_linear), '{0:.4f}'.format(L_geodesic)]
    
    G_plot = G_plot.to('cpu')
    torch.save({'G_plot': gamma_g_geodesic,
                'arc_length': arc_length,
                'T': args.T}, 
               args.save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    