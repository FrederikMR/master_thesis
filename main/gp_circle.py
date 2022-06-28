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

import matplotlib.pyplot as plt
import matplotlib as mpl

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d
import rm
import rm_param as pm
import gaussian_proces as gp
import sp
import kernels as km

#%% Function for plotting

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

#%% Load data and set hyper-parameters

N_training = 100
N_sim = 10

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

sigman = 0.1
noise = sp.sim_multinormal(mu=jnp.zeros(3), cov=(sigman**2)*jnp.eye(3), dim=N_training).reshape(N_training, 3)

#sigman = 0.0
#noise = jnp.zeros((N_training, 3))

theta = jnp.linspace(0,2*jnp.pi,N_training)

x1 = jnp.cos(theta)+noise[:,0]
x2 = jnp.sin(theta)+noise[:,1]
y_plot = noise[:,2]

theta = jnp.linspace(0,2*jnp.pi, 1000)
x1_plot = jnp.cos(theta)
x2_plot = jnp.sin(theta)
x3_plot = jnp.zeros_like(theta)

X_training = jnp.vstack((x1,x2))

y_training = jnp.vstack((x1,x2,y_plot.reshape(-1)))

#%% Set up GP and analytical RM

data_plot = plot_3d_fun(N=100)
RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=10, tol=0.01, method='euler', grid=jnp.linspace(0,1,100))
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=10, tol=0.01, method='euler', grid=jnp.linspace(0,1,100))

#%% Plot data and reconstructed (mean) manifold

plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('z')
ax.set_title('Simulated Data')
        
ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black')
ax.plot(x1_plot, x2_plot, x3_plot, color='blue')
        
plt.tight_layout()

plt.show()

rec_data, _ = RMEG.post_mom(X_training)
plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('z')
ax.set_title('Reconstructed Data')
        
ax.scatter3D(rec_data[0], rec_data[1], rec_data[2], color='black')
ax.plot(x1_plot, x2_plot, x3_plot, color='blue')
        
plt.tight_layout()

plt.show()

#%% Curvature

n_points = 100
x = jnp.linspace(-2.0,2.0,n_points)
y = jnp.linspace(-2.0,2.0,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.transpose(jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points))), axes=(1,2,0))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

secEG = vmap(lambda y1: vmap(lambda y2: RMEG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
ax.plot_surface(X[:,:,0], X[:,:,1], secEG, color='orange', alpha=0.5)


for i in range(N_sim):
    secSG = vmap(lambda y1: vmap(lambda y2: RMSG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), eps[i]))(y1))(X)
    ax.plot_surface(X[:,:,0], X[:,:,1], secSG, color='cyan', alpha=0.2)

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=1.0)


idx_max = jnp.where(secSG==secSG.max())
test = RMSG.DDG(X[idx_max[0],idx_max[1]], eps[0])

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
legend3 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
ax.legend([legend1, legend2, legend3], ['Realised Curvature', 'Curvature for EG', 'True Curvature'], numpoints = 1)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X[:,:,0], X[:,:,1], secEG, color='orange', alpha=0.2)
ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=1.0)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
ax.legend([legend2], ['Curvature for EG'], numpoints = 1)

plt.tight_layout()
plt.show()

#%% IVP plot

theta0 = jnp.pi/3

x0 = jnp.array([jnp.cos(theta0), jnp.sin(theta0)])
v0 = jnp.array([0.5, -0.50])

gammaEG, _ = RMEG.geo_ivp(x0,v0)
gammaEG_manifold, _ = RMEG.post_mom(gammaEG.T)

gammaSG = vmap(lambda x: RMSG.geo_ivp(x0, v0, x)[0])(eps)

gammaSG_plot = []
for i in range(N_sim):
    if jnp.max(gammaSG[i])<2:
        gammaSG_plot.append(gammaSG[i])
if gammaSG_plot:
    gammaSG_plot = jnp.stack(gammaSG_plot)

gammaSG_manifold, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gammaSG_plot))

plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")
low_val = 1

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)
ax.plot(x1_plot, x2_plot, x3_plot, color='blue')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
        
plt.tight_layout()

plt.show()


plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")
low_val = 1

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')

for i in range(N_sim-1):
    x = gammaSG_manifold[i].T[:,0]
    y = gammaSG_manifold[i].T[:,1]
    z = gammaSG_manifold[i].T[:,2]
    ax.plot(x, y, z, color='cyan')

x = gammaSG_manifold[-1].T[:,0]
y = gammaSG_manifold[-1].T[:,1]
z = gammaSG_manifold[-1].T[:,2]
ax.plot(x, y, z, label='GP Geodesic for SG', color='cyan')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)
ax.plot(x1_plot, x2_plot, x3_plot, color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
        
plt.tight_layout()

plt.show()

plt.figure(figsize=(8,6))

x = gammaEG[:,0]
y = gammaEG[:,1]
plt.plot(x, y, '-*', label='GP Geoodesic for EG', color='orange')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Geodesic in Z')

plt.tight_layout()

plt.show()

plt.figure(figsize=(8,6))

x = gammaEG[:,0]
y = gammaEG[:,1]
plt.plot(x, y, '-*', label='GP Geoodesic for EG', color='orange')

for i in range(N_sim-1):
    x = gammaSG_plot[i][:,0]
    y = gammaSG_plot[i][:,1]
    plt.plot(x, y, color='cyan')

x = gammaSG_plot[-1][:,0]
y = gammaSG_plot[-1][:,1]
plt.plot(x, y, label='GP Geodesic for SG', color='cyan')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Geodesic in Z')

plt.tight_layout()


plt.show()

#%% BVP plot

theta0 = jnp.pi/2
thetaT = 0.0

x0 = jnp.array([jnp.cos(theta0), jnp.sin(theta0)])
xT = jnp.array([jnp.cos(thetaT), jnp.sin(thetaT)])

gammaEG, _ = RMEG.geo_bvp(x0,xT)
gammaEG_manifold, _ = RMEG.post_mom(gammaEG.T)

gammaSG_plot = []
for i in range(N_sim):
    gammaSG, _ = RMSG.geo_bvp(x0, xT, eps[i])
    if jnp.max(gammaSG)<2:
        gammaSG_plot.append(gammaSG)
if gammaSG_plot:
    gammaSG_plot = jnp.stack(gammaSG_plot)

gammaSG_manifold, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gammaSG_plot))

load_path = 'rm_computations/circle'
load = load_path+'.pt'
checkpoint = torch.load(load, map_location='cpu')
gx = checkpoint['gx']
gy = checkpoint['gy']
points = torch.transpose(torch.cat((gx.view(-1,1), gy.view(-1,1)), dim = 1), 0, 1)

G_old = checkpoint['G_old'].detach().numpy()
G_new = checkpoint['G_new'].detach().numpy()
L_old = checkpoint['L_old']
L_new = checkpoint['L_new']
z_linear = checkpoint['z_linear'].detach().numpy()
z_geodesic = checkpoint['z_geodesic'].detach().numpy()

plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")
low_val = 1

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')
ax.plot(G_new[:,0], G_new[:,1], G_new[:,2], label='VAE Geodesic', color='purple')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)
ax.plot(x1_plot, x2_plot, x3_plot, color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
        
plt.tight_layout()

plt.show()


plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")
low_val = 1

x1_grid = jnp.linspace(-2.0, 2.0, num = 100)
x2_grid = jnp.linspace(-2.0, 2.0, num = 100)

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')
ax.plot(G_new[:,0], G_new[:,1], G_new[:,2], label='VAE Geodesic', color='purple')

for i in range(N_sim-1):
    x = gammaSG_manifold[i].T[:,0]
    y = gammaSG_manifold[i].T[:,1]
    z = gammaSG_manifold[i].T[:,2]
    ax.plot(x, y, z, color='cyan')

x = gammaSG_manifold[-1].T[:,0]
y = gammaSG_manifold[-1].T[:,1]
z = gammaSG_manifold[-1].T[:,2]
ax.plot(x, y, z, label='GP Geodesic for SG', color='cyan')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)
ax.plot(x1_plot, x2_plot, x3_plot, color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
        
plt.tight_layout()

plt.show()

plt.figure(figsize=(8,6))

x = gammaEG[:,0]
y = gammaEG[:,1]
plt.plot(x, y, '-*', label='GP Geoodesic for EG', color='orange')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Geodesic in Z')

plt.tight_layout()


plt.show()


plt.figure(figsize=(8,6))

x = gammaEG[:,0]
y = gammaEG[:,1]
plt.plot(x, y, '-*', label='GP Geoodesic for EG', color='orange')


for i in range(N_sim-1):
    x = gammaSG_plot[i][:,0]
    y = gammaSG_plot[i][:,1]
    plt.plot(x, y, color='cyan')

x = gammaSG_plot[-1][:,0]
y = gammaSG_plot[-1][:,1]
plt.plot(x, y, label='GP Geodesic for SG', color='cyan')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Geodesic in Z')

plt.tight_layout()


plt.show()
