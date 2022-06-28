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

def fun(x1, x2):
    
    return jnp.array([x1, x2, x1**2+x2**2])

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

N_training = 10
N_sim = 10
sigman = 0.0

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 1.5, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
y_plot = x1_mesh**2+x2_mesh**2

x = jnp.linspace(-2,2,100)
y = jnp.linspace(-2,2,100)
X1, X2 = jnp.meshgrid(x, y)
Z = X1**2+X2**2

x1 = x1_mesh.reshape(-1)
x2 = x2_mesh.reshape(-1)

X_training = jnp.vstack((x1,x2))

y_training = jnp.vstack((x1,x2,y_plot.reshape(-1)))

#%% Set up GP and analytical RM

data_plot = plot_3d_fun(N=100)
RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=10, tol=0.01, method='euler', grid=jnp.linspace(0,1,100))
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=10, tol=0.01, method='euler', grid=jnp.linspace(0,1,100))
RM = rm.rm_geometry(param_fun=lambda x: jnp.array([x[0], x[1], x[0]**2+x[1]**2]), method='euler')

#%% Plot data and reconstructed (mean) manifold

#Plotting the raw data
data_plot.plot_data_scatter_3d(fun, y_training[0], y_training[1], y_training[2]) #Plotting the true surface with the simulated data

rec_data, _ = RMEG.post_mom(X_training)
data_plot.plot_data_scatter_3d(fun, rec_data[0], rec_data[1], rec_data[2], title='Scatter of Reconstructed Data')

#%% Curvature

n_points = 10
x = jnp.linspace(-2.0,2.0,n_points)
y = jnp.linspace(-2.0,2.0,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.transpose(jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points))), axes=(1,2,0))

print(RMSG.DDcov(jnp.array([1.0,1.0])))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

secEG = vmap(lambda y1: vmap(lambda y2: RMEG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
ax.plot_surface(X[:,:,0], X[:,:,1], secEG, color='orange', alpha=0.5)

sectrue = vmap(lambda y1: vmap(lambda y2: RM.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
ax.plot_surface(X[:,:,0], X[:,:,1], sectrue, color='red', alpha=0.8)

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
ax.plot_surface(X[:,:,0], X[:,:,1], sectrue, color='red', alpha=0.8)
ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=1.0)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
legend3 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
ax.legend([legend2, legend3], ['Curvature for EG', 'True Curvature'], numpoints = 1)

plt.tight_layout()
plt.show()



#%% IVP plot

x0 = jnp.array([0.5, -1.0])
v0 = jnp.array([0.5, 0.5])

gammaEG, _ = RMEG.geo_ivp(x0,v0)
gammaEG_manifold, _ = RMEG.post_mom(gammaEG.T)

gamma_true, _ = RM.geo_ivp(x0, v0)
gamma_true_manifold = fun(gamma_true[:,0], gamma_true[:,1])

gammaSG = vmap(lambda x: RMSG.geo_ivp(x0, v0, x)[0])(eps)

gammaSG_plot = []
for i in range(N_sim):
    if jnp.max(gammaSG[i])<10:
        gammaSG_plot.append(gammaSG[i])
if gammaSG_plot:
    gammaSG_plot = jnp.stack(gammaSG_plot)

gammaSG_manifold, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gammaSG_plot))

plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")
low_val = 1

x1_grid = jnp.linspace(-2.0, 2.0, num = 100)
x2_grid = jnp.linspace(-2.0, 2.0, num = 100)

X1, X2 = np.meshgrid(x1_grid, x2_grid)
X1, X2, X3 = fun(X1, X2)
ax.plot_surface(
X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')

x = gamma_true_manifold.T[:, 0]
y = gamma_true_manifold.T[:, 1]
z = gamma_true_manifold.T[:, 2]
ax.plot(x, y, z, label='True Geodesic', color='red')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)

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

X1, X2 = np.meshgrid(x1_grid, x2_grid)
X1, X2, X3 = fun(X1, X2)
ax.plot_surface(
X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')

x = gamma_true_manifold.T[:, 0]
y = gamma_true_manifold.T[:, 1]
z = gamma_true_manifold.T[:, 2]
ax.plot(x, y, z, label='True Geodesic', color='red')

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

x = gamma_true[:,0]
y = gamma_true[:,1]
plt.plot(x, y, '-*', label='True Geoodesic', color='red')

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

x = gamma_true[:,0]
y = gamma_true[:,1]
plt.plot(x, y, '-*', label='True Geoodesic', color='red')


for i in range(N_sim-1):
    x = gammaSG_plot[i][:,0]
    y = gammaSG_plot[i][:,1]
    plt.plot(x, y, color='orange')

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

x0 = jnp.array([0.5, -1.0])
xT = jnp.array([0.5, 0.5])

gammaEG, _ = RMEG.geo_bvp(x0,xT)
gammaEG_manifold, _ = RMEG.post_mom(gammaEG.T)

gamma_true, _ = RM.geo_bvp(x0, xT)
gamma_true_manifold = fun(gamma_true[:,0], gamma_true[:,1])

gammaSG = vmap(lambda x: RMSG.geo_ivp(x0, v0, x)[0])(eps)


gammaSG_plot = []
for i in range(N_sim):
    gammaSG, _ = RMSG.geo_bvp(x0, xT, eps[i])
    if jnp.max(gammaSG)<2:
        gammaSG_plot.append(gammaSG)
if gammaSG_plot:
    gammaSG_plot = jnp.stack(gammaSG_plot)

gammaSG_manifold, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gammaSG_plot))

load_path = 'rm_computations/paraboloid_simple_geodesic.pt'

checkpoint = torch.load(load_path, map_location='cpu')
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

plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")
low_val = 1

x1_grid = jnp.linspace(-2.0, 2.0, num = 100)
x2_grid = jnp.linspace(-2.0, 2.0, num = 100)

X1, X2 = np.meshgrid(x1_grid, x2_grid)
X1, X2, X3 = fun(X1, X2)
ax.plot_surface(
X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')
ax.plot(g_geodesic[:,0], g_geodesic[:,1], g_geodesic[:,2], label='VAE Geodesic', color='purple')

x = gamma_true_manifold.T[:, 0]
y = gamma_true_manifold.T[:, 1]
z = gamma_true_manifold.T[:, 2]
ax.plot(x, y, z, label='True Geodesic', color='red')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)

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

X1, X2 = np.meshgrid(x1_grid, x2_grid)
X1, X2, X3 = fun(X1, X2)
ax.plot_surface(
X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

x = gammaEG_manifold.T[:,0]
y = gammaEG_manifold.T[:,1]
z = gammaEG_manifold.T[:,2]
ax.plot(x, y, z, label='GP Geodesic for EG', color='orange')
ax.plot(g_geodesic[:,0], g_geodesic[:,1], g_geodesic[:,2], label='VAE Geodesic', color='purple')

x = gamma_true_manifold.T[:, 0]
y = gamma_true_manifold.T[:, 1]
z = gamma_true_manifold.T[:, 2]
ax.plot(x, y, z, label='True Geodesic', color='red')

for i in range(N_sim-1):
    x = gammaSG_manifold[i].T[:,0]
    y = gammaSG_manifold[i].T[:,1]
    z = gammaSG_manifold[i].T[:,2]
    ax.plot(x, y, z, color='orange')

x = gammaSG_manifold[-1].T[:,0]
y = gammaSG_manifold[-1].T[:,1]
z = gammaSG_manifold[-1].T[:,2]
ax.plot(x, y, z, label='GP Geodesic for SG', color='cyan')

ax.scatter3D(y_training[0], y_training[1], y_training[2], color='black', alpha=0.2)

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

x = gamma_true[:,0]
y = gamma_true[:,1]
plt.plot(x, y, '-*', label='True Geoodesic', color='red')

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

x = gamma_true[:,0]
y = gamma_true[:,1]
plt.plot(x, y, '-*', label='True Geoodesic', color='red')


for i in range(N_sim-1):
    x = gammaSG_plot[i][:,0]
    y = gammaSG_plot[i][:,1]
    plt.plot(x, y, color='orange')

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
