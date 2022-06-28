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

#Own files
from plot_dat import plot_3d_fun
from AE_surface3d import AE_3d
#%% Function for plotting

def fun_para(x1, x2):
    
    return x1, x2, x1**2+x2**2

def fun_plane(x1, x2):
    
    return x1, x2, 0*x1

#%% Loading data and model

data_name = 'paraboloid'

if data_name == 'paraboloid':
    fun = fun_para
elif data_name == 'plane':
    fun = fun_plane


#Hyper-parameters
epoch_load = '100000'
lr = 0.0001
device = 'cpu'

#Loading files
data_path = 'sim_data/'+data_name+'.csv'
file_model_save = 'trained_models/AE_'+data_name+'_epoch_'+epoch_load+'.pt'
data_plot = plot_3d_fun(N=100)

#Loading data
df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Loading model
model = AE_3d().to(device) #Model used
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
rec_loss = checkpoint['rec_loss']

model.eval()

#%% Plotting true data

#Plotting the raw data
x1 = DATA[:,0].detach().numpy()
x2 = DATA[:,1].detach().numpy()
x3 = DATA[:,2].detach().numpy()
data_plot.true_Surface3d(fun, [min(x1), max(x1)], [min(x2), max(x2)]) #Plotting the true surface
data_plot.plot_data_scatter_3d(fun, x1, x2, x3) #Plotting the true surface with the simulated data
data_plot.plot_data_surface_3d(x1, x2, x3) #Surface plot of the data

#%% Plotting learned data

X = model(DATA) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
z = X[0]
x_hat = X[1]
x1 = x_hat[:,0].detach().numpy()
x2 = x_hat[:,1].detach().numpy()
x3 = x_hat[:,2].detach().numpy()

#Plotting loss function
data_plot.plot_loss(rec_loss, title='Reconstruction Loss')

#Surface plot of the reconstructed data
data_plot.plot_data_surface_3d(x1, x2, x3)

#Plotting the true surface with the reconstructed data
data_plot.plot_data_scatter_3d(fun, x1, x2, x3, title='Scatter of Reconstructed Data')

#Plotting mu in Z-space
z = z.detach().numpy()
data_plot.plot_dat_in_Z_2d([z, 'z'])

#%% Circle

def circle_fun(N = 1000, mu = np.array([1.,1.,1.]), r=1):
    
    theta = np.linspace(0, 2*np.pi, N)
    x1 = r*np.cos(theta)+mu[0]
    x2 = r*np.sin(theta)+mu[1]
    x3 = np.zeros(N)+mu[2]
    
    return x1, x2, x3

#%% Loading data and model

#Hyper-parameters
epoch_load = '10000'
lr = 0.0001
device = 'cpu'
latent_dim = 2

#Data
data_name = 'circle'
fun = circle_fun

#Loading files
data_path = 'sim_data/'+data_name+'.csv'
file_model_save = 'trained_models/AE_'+data_name+'_epoch_'+epoch_load+'.pt'
data_plot = plot_3d_fun(N=100)

#Loading data
df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Loading model
model = AE_3d().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
rec_loss = checkpoint['rec_loss']

model.eval()

#%% Plotting true data

#Plotting the raw data
X = DATA.detach().numpy()
data_plot.true_path3d_with_points(fun, X, [0, 2], [0,2], [0,2]) #Plotting the true surface

#%% Plotting learned data

X = model(DATA) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
z = X[0]
x_hat = X[1].detach().numpy()

#Plotting loss function
data_plot.plot_loss(rec_loss, title='Reconstruction Loss')

data_plot.true_path3d_with_points(fun, x_hat, [0, 2], [0,2], [0,2]) #Plotting the true surface

#Plotting mu in Z-space
z = z.detach().numpy()
data_plot.plot_dat_in_Z_2d([z, 'z'])






























