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
import pandas as pd
import numpy as np
import jax.numpy as jnp

import jax.numpy as jnp
from jax import vmap, jacfwd

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d
import rm
import rm_param as pm
import gaussian_proces as gp
import sp
import kernels as km

#%% Function for plotting

def fun_para(x1, x2):
    
    return x1, x2, x1**2+x2**2

def fun_plane(x1, x2):
    
    return x1, x2, 0*x1

def k_fun(x,y, beta=1.0, omega=1.0):
    
    x_diff = x-y
    
    return beta*jnp.exp(-omega*jnp.dot(x_diff, x_diff)/2)

def Dk_fun(x,y, beta=1.0, omega=1.0):
    
    x_diff = y-x
    
    return omega*x_diff*k_fun(x,y,beta,omega)

def DDk_fun(x,y, beta=1.0, omega=1.0):
    
    N = len(x)
    x_diff = (x-y).reshape(1,-1)
    
    return -omega*k_fun(x,y,beta,omega)*(x_diff.T.dot(x_diff)*omega-jnp.eye(N))

#%% Loading data and model

data_name = 'plane'

if data_name == 'paraboloid':
    fun = fun_para
    jax_fun = lambda x: pm.simple_polynomial(x, jnp.array([1.0,1.0]), n=2)
elif data_name == 'plane':
    fun = fun_plane
    jax_fun = lambda x: pm.plane(x, jnp.array([0.0,0.0]))

RM = rm.rm_geometry(param_fun = jax_fun)

#Hyper-parameters
epoch_load = '100000'
lr = 0.0001
device = 'cpu'

#Loading files
data_path = 'sim_data/'+data_name+'.csv'
file_model_save = 'trained_models/VAE_'+data_name+'_epoch_'+epoch_load+'.pt'
data_plot = plot_3d_fun(N=100)

#Loading data
df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Loading model
model = model = VAE_3d().to(device) #Model used
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
elbo = checkpoint['ELBO']
rec_loss = checkpoint['rec_loss']
kld_loss = checkpoint['KLD']

model.eval()

#%% Plotting true data

#Plotting the raw data
x1 = DATA[:,0].detach().numpy()
x2 = DATA[:,1].detach().numpy()
x3 = DATA[:,2].detach().numpy()
data_plot.true_Surface3d(fun, [min(x1), max(x1)], [min(x2), max(x2)]) #Plotting the true surface
data_plot.plot_data_scatter_3d(fun, x1, x2, x3) #Plotting the true surface with the simulated data
data_plot.plot_data_surface_3d(x1, x2, x3) #Surface plot of the data

#%% Learning GP

N_sim = 10

X_training = jnp.vstack((jnp.asarray(x1), jnp.asarray(x2)))
y_training = jnp.vstack((jnp.asarray(x1), jnp.asarray(x2),jnp.asarray(x3)))
sigman = 0.0

RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)

#%% Reconstructed data

rec_data, _ = RMEG.post_mom(X_training)
data_plot.plot_data_scatter_3d(fun, rec_data[0], rec_data[1], rec_data[2], title='Scatter of Reconstructed Data')

#%% BVP Geodesics EG

load_path = 'rm_computations/'+data_name+'_simple_geodesic.pt'

checkpoint = torch.load(load_path, map_location=device)
g_geodesic = checkpoint['g_geodesic']
gx = g_geodesic[0]
gy = g_geodesic[-1]
points = torch.transpose(torch.cat((gx.view(-1,1), gy.view(-1,1)), dim = 1), 0, 1)

z_linear = checkpoint['z_linear'].detach().numpy()
z_geodesic = checkpoint['z_geodesic'].detach().numpy()
g_linear = checkpoint['g_linear'].detach().numpy()
g_geodesic = g_geodesic.detach().numpy()
L_linear = checkpoint['L_linear']
L_geodesic = checkpoint['L_geodesic']

z0 = jnp.array([-1.0, -1.0])
zT = jnp.array([1.0, -1.0])

z_true_geodesic, _ = RM.geo_bvp(z0,zT)

z_true_geodesic = z_true_geodesic
g1, g2, g3 = fun(z_true_geodesic[:,0], z_true_geodesic[:,1])
g_true_geodesic = np.vstack((g1,g2,g3)).transpose()

gamma, _ = RMEG.geo_bvp(z0, zT)
gamma_manifold, _ = RMEG.post_mom(gamma.T)

data_plot.plot_geodesic_in_X_3d(fun, #points.detach().numpy(),
                          [-4,4],[-4,4],
                          [gamma_manifold.T, 'GP Geodesic'], 
                          [g_geodesic, 'VAE Geodesic'],
                          [g_true_geodesic, 'True Geodesic'])

data_plot.plot_geodesic_in_Z_2d([gamma, 'GP Geodesic'], 
                          [z_true_geodesic, 'True Geodesic'])

#%% BVP Geodesic SG

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

gamma2_plot = []
for i in range(N_sim):
    gamma2, _ = RMSG.geo_bvp(z0, zT, eps[i])
    if jnp.max(gamma2[i])<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))

#%% IVP Geodesic SG

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

gamma2 = vmap(lambda x: RMSG.geo_ivp(z0, zT*0.1, x)[0])(eps)

gamma2_plot = []
for i in range(N_sim):
    if jnp.max(gamma2[i])<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))

#%% Plotting learned data

X = model(DATA) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
z = X[0]
x_hat = X[1]
mu = X[2]
std = X[3]
x1 = x_hat[:,0].detach().numpy()
x2 = x_hat[:,1].detach().numpy()
x3 = x_hat[:,2].detach().numpy()

#Plotting loss function
data_plot.plot_loss(elbo, title='Loss function')
data_plot.plot_loss(rec_loss, title='Reconstruction Loss')
data_plot.plot_loss(kld_loss, title='KLD')

#Surface plot of the reconstructed data
data_plot.plot_data_surface_3d(x1, x2, x3)

#Plotting the true surface with the reconstructed data
data_plot.plot_data_scatter_3d(fun, x1, x2, x3, title='Scatter of Reconstructed Data')

#Plotting mu in Z-space
z = z.detach().numpy()
mu = mu.detach().numpy()
std = std.detach().numpy()
data_plot.plot_dat_in_Z_2d([z, 'z'])
data_plot.plot_dat_in_Z_2d([mu, 'mu'])
data_plot.plot_dat_in_Z_2d([std, 'std'])

#%% Plotting the Riemannian simple geodesics

load_path = 'rm_computations/'+data_name+'_simple_geodesic.pt'

checkpoint = torch.load(load_path, map_location=device)
g_geodesic = checkpoint['g_geodesic']
gx = g_geodesic[0]
gy = g_geodesic[-1]
points = torch.transpose(torch.cat((gx.view(-1,1), gy.view(-1,1)), dim = 1), 0, 1)

z_linear = checkpoint['z_linear'].detach().numpy()
z_geodesic = checkpoint['z_geodesic'].detach().numpy()
g_linear = checkpoint['g_linear'].detach().numpy()
g_geodesic = g_geodesic.detach().numpy()
L_linear = checkpoint['L_linear']
L_geodesic = checkpoint['L_geodesic']

zx = np.array([-1.0,-1.0])
zy = np.array([1.0,-1.0])

z_true_geodesic, _ = RM.geo_bvp(zx,zy)

z_true_geodesic = z_true_geodesic
g1, g2, g3 = fun(z_true_geodesic[:,0], z_true_geodesic[:,1])
g_true_geodesic = np.vstack((g1,g2,g3)).transpose()

data_plot.plot_geodesic_in_X_3d(fun, #points.detach().numpy(),
                          [-4,4],[-4,4],
                          [g_linear, 'Linear Interpolation'], 
                          [g_geodesic, 'Approximated Geodesic'],
                          [g_true_geodesic, 'True Geodesic'])

data_plot.plot_geodesic_in_Z_2d([z_linear, 'Linear Interpolation'], 
                          [z_geodesic, 'Approximated Geodesic'])