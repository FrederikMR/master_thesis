#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 18:18:33 2022

@author: frederik
"""

#%% Sources

#%% Modules

#Adds the source library to Python Path
import sys
sys.path.insert(1, '../src/')

import jax.numpy as jnp
from jax import vmap

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib as mpl

#Own modules
import rm_param as pm
import sp
import rm

#%% Set hyper parameters

N_sim = 10
n_points = 100
n_grid = 100

a = 2
b = 2

grid0x1 = 0
grid1x1 = a
grid0x2 = 0
grid1x2 = b

y0 = jnp.array([0.5,0.5])
yT = jnp.array([1.5,1.0])

N_k = 2
N_l = 2
n_coef = N_k*N_l
mu = jnp.zeros(n_coef)
sigma = []
for k in range(N_k):
    for l in range(N_l):
        sigma.append(1/((l+1)*(k+1))**5)
sigma = jnp.stack(sigma)
cov = jnp.diag(sigma)
coef = (sigma*sp.sim_multinormal(mu, jnp.eye(N_k*N_l), dim=N_sim)).reshape(N_sim, N_k, N_l, order='C')

#%% Set basis functions

def fkl(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return jnp.sin(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)

def fkl_x1(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return k*jnp.pi*jnp.cos(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)/a

def fkl_x2(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return l*jnp.pi*jnp.sin(k*jnp.pi*x1/a)*jnp.cos(l*jnp.pi*x2/b)/b

def fkl_x1x1(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return -((k*jnp.pi)**2)*jnp.sin(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)/(a**2)

def fkl_x1x2(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return k*l*(jnp.pi**2)*jnp.cos(k*jnp.pi*x1/a)*jnp.cos(l*jnp.pi*x2/b)/(a*b)

def fkl_x2x2(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return -((l*jnp.pi)**2)*jnp.sin(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)/(b**2)

def fkl_x1x1x1(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return -((k*jnp.pi)**3)*jnp.cos(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)/(a**3)

def fkl_x1x1x2(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return -(k**2)*l*(jnp.pi**3)*jnp.sin(k*jnp.pi*x1/a)*jnp.cos(l*jnp.pi*x2/b)/((a**2)*b)

def fkl_x2x2x1(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return -(l**2)*k*(jnp.pi**3)*jnp.cos(k*jnp.pi*x1/a)*jnp.sin(l*jnp.pi*x2/b)/((b**2)*a)

def fkl_x2x2x2(x,k,l):
    
    x1 = x[0]
    x2 = x[1]
    
    return -((l*jnp.pi)**3)*jnp.sin(k*jnp.pi*x1/a)*jnp.cos(l*jnp.pi*x2/b)/(b**3)

f_fun = fkl
Df_fun = [fkl_x1, fkl_x2]
DDf_fun = [fkl_x1x1, fkl_x1x2, fkl_x2x2]
DDDf_fun = [fkl_x1x1x1, fkl_x1x1x2, fkl_x2x2x1, fkl_x2x2x2]

#%% Manifold parametrization 

x = jnp.linspace(grid0x1,grid1x1,n_points)
y = jnp.linspace(grid0x2,grid1x2,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
X = jnp.einsum('ijk->jki', X)

param_fun = lambda x, coef: pm.basis_rec(x, coef, a=a, b=b, N_k=N_k, N_l=N_l)
    
#%% Plot realisations of the stochastic manifolds

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i in range(N_sim):
    Z = vmap(lambda y: vmap(lambda x: param_fun(x,coef[i]))(y))(X)
    ax.plot_surface(Z[:,:,0], Z[:,:,1], Z[:,:,2], color='cyan', alpha=0.2)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$f(x_{1},x_{2})$')
ax.set_title('Realisations of stochastic Riemannian manifold')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realisations'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Define manifold structure

cov = (jnp.eye(N_k*N_l)*sigma).reshape(N_k,N_l,N_k,N_l)
RMSG = rm.rm_2dbasisSG(f_fun, Df_fun, DDf_fun, DDDf_fun, N_k=N_k, N_l=N_l, n_steps=100)
RMEG = rm.rm_2dbasisEG(f_fun, mu.reshape(N_k,N_l), cov, Df_fun, DDf_fun, DDDf_fun, N_k=N_k, N_l=N_l, n_steps=100)

x = jnp.array([0.5, 0.5])

#%% Curvature

N_sample = 100
coef_sample = (sigma*sp.sim_multinormal(mu, jnp.eye(N_k*N_l), dim=N_sample)).reshape(N_sample, N_k, N_l, order='C')
sec = vmap(lambda co: vmap(lambda y1: vmap(lambda y2: RMSG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), co))(y1))(X))(coef_sample)
sec_samplemean = jnp.mean(sec, axis=0)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

sec = vmap(lambda co: vmap(lambda y1: vmap(lambda y2: RMSG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), co))(y1))(X))(coef)
for i in range(N_sim):
    #sec = vmap(lambda y1: vmap(lambda y2: rm.rm_geometry(param_fun=lambda x: param_fun(x, coef[i])).SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
    ax.plot_surface(X[:,:,0], X[:,:,1], sec[i], color='cyan', alpha=0.2)

secEG = vmap(lambda y1: vmap(lambda y2: RMEG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])))(y1))(X)
ax.plot_surface(X[:,:,0], X[:,:,1], secEG, color='red', alpha=1.0)

ax.plot_surface(X[:,:,0], X[:,:,1], sec_samplemean, color='black', alpha=0.8)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
#ax.axes.set_zlim3d(bottom=-1, top=1)
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend3 = mpl.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
ax.legend([legend1, legend2, legend3], ['Realised Curvature', 'Curvature for Expected metric', 'Sample Mean Curvature'], numpoints = 1)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X[:,:,0], X[:,:,1], secEG, color='red', alpha=0.2)
ax.plot_surface(X[:,:,0], X[:,:,1], sec_samplemean, color='black', alpha=0.5)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Sectional Curvature')
#ax.axes.set_zlim3d(bottom=-1, top=1)
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend3 = mpl.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
ax.legend([legend2, legend3], ['Curvature for Expected metric', 'Sample Mean Curvature'], numpoints = 1)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X[:,:,0], X[:,:,1], secEG-sec_samplemean, color='red', alpha=1.0)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Difference between Sample Mean Curvature and Curvature for Expected Metric')

plt.tight_layout()
plt.show()


#%% Geodesics

fig, ax = plt.subplots(1,2,figsize=(8,6))

y0 = jnp.array([1.0,1.0])
v0 = jnp.array([-0.5, 0.5])

N_sample = 100
coef_sample = (sigma*sp.sim_multinormal(mu, jnp.eye(N_k*N_l), dim=N_sample)).reshape(N_sample, N_k, N_l, order='C')
gamma = vmap(lambda co: RMSG.geo_ivp(y0,v0, co)[0])(coef_sample)
gamma_samplemean = jnp.mean(gamma, axis=0)

for i in range(N_sim):
    ax[0].plot(jnp.linspace(0,1,gamma[i].shape[0]), gamma[i][:,0], color='cyan', alpha=0.4)
    ax[1].plot(jnp.linspace(0,1,gamma[i].shape[0]), gamma[i][:,1], color='cyan', alpha=0.4)

gammaEG, _ = RMEG.geo_ivp(y0,v0)
#gamma, _ = rm.rm_geometry(param_fun=lambda x: param_fun(x, coef[i])).geo_ivp(y0,v0)
ax[0].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,0], color='red', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,1], color='red', alpha=1.0)

ax[0].plot(jnp.linspace(0,1,gammaEG.shape[0]), gamma_samplemean[:,0], color='black', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gammaEG.shape[0]), gamma_samplemean[:,1], color='black', alpha=1.0)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()

legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend3 = mpl.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
ax[0].legend([legend1, legend2, legend3], [r'$\gamma^{1}(t)$', r'$\gamma^{1}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$', r'$\mathbb{E}\left[\gamma^{1}(t)\right]$'], numpoints = 1)
ax[1].legend([legend1, legend2, legend3], [r'$\gamma^{2}(t)$', r'$\gamma^{2}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$', r'$\mathbb{E}\left[\gamma^{2}(t)\right]$'], numpoints = 1)


fig, ax = plt.subplots(1,2,figsize=(8,6))

ax[0].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,0]-gamma_samplemean[:,0], color='red', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,1]-gamma_samplemean[:,1], color='red', alpha=1.0)

fig.suptitle('Error between Sample Mean Geodesic and Geodesic for Expected Metric')
ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\mathbb{E}\left[\gamma^{1}(t)\right]-\gamma^{1}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\mathbb{E}\left[\gamma^{2}(t)\right]-\gamma^{2}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$')
ax[0].grid()
ax[1].grid()

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')

Z_gamma = vmap(lambda co: vmap(lambda x: param_fun(x, co))(gamma[i]))(coef)
Z_param = vmap(lambda co: vmap(lambda y: vmap(lambda x: param_fun(x,co))(y))(X))(coef)
for i in range(N_sim):
    ax.plot(Z_gamma[i][:,0], Z_gamma[i][:,1], Z_gamma[i][:,2], color='red', alpha=1.0, linewidth=3.0)
    ax.plot_surface(Z_param[i][:,:,0], Z_param[i][:,:,1], Z_param[i][:,:,2], color='cyan', alpha=0.1)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$\gamma$')
ax.set_title('Realisations of stochastic geodesics')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
ax.legend([legend], [r'$\gamma(t)$'], numpoints = 1)

#%% Geodesics

fig, ax = plt.subplots(1,2,figsize=(8,6))

x0 = jnp.array([1.0,1.0])
xT = jnp.array([0.0, 0.0])

N_sample = 100
coef_sample = (sigma*sp.sim_multinormal(mu, jnp.eye(N_k*N_l), dim=N_sample)).reshape(N_sample, N_k, N_l, order='C')
gamma = []
for i in range(N_sample):
    print(i)
    gamma.append(RMSG.geo_bvp(x0,xT, coef_sample[i])[0])
gamma = jnp.stack(gamma)

gamma_samplemean = jnp.mean(gamma, axis=0)

gamma = []
for i in range(N_sim):
    gamma.append(RMSG.geo_bvp(x0,xT, coef[i])[0])
gamma = jnp.stack(gamma)

for i in range(N_sim):
    ax[0].plot(jnp.linspace(0,1,gamma[i].shape[0]), gamma[i][:,0], color='cyan', alpha=0.4)
    ax[1].plot(jnp.linspace(0,1,gamma[i].shape[0]), gamma[i][:,1], color='cyan', alpha=0.4)

gammaEG, _ = RMEG.geo_bvp(x0,xT)
#gamma, _ = rm.rm_geometry(param_fun=lambda x: param_fun(x, coef[i])).geo_ivp(y0,v0)
ax[0].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,0], color='red', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,1], color='red', alpha=1.0)

ax[0].plot(jnp.linspace(0,1,gammaEG.shape[0]), gamma_samplemean[:,0], color='black', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gammaEG.shape[0]), gamma_samplemean[:,1], color='black', alpha=1.0)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()

legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend3 = mpl.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
ax[0].legend([legend1, legend2, legend3], [r'$\gamma^{1}(t)$', r'$\gamma^{1}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$', r'$\mathbb{E}\left[\gamma^{1}(t)\right]$'], numpoints = 1)
ax[1].legend([legend1, legend2, legend3], [r'$\gamma^{2}(t)$', r'$\gamma^{2}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$', r'$\mathbb{E}\left[\gamma^{2}(t)\right]$'], numpoints = 1)


fig, ax = plt.subplots(1,2,figsize=(8,6))

ax[0].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,0]-gamma_samplemean[:,0], color='red', alpha=1.0)
ax[1].plot(jnp.linspace(0,1,gammaEG.shape[0]), gammaEG[:,1]-gamma_samplemean[:,1], color='red', alpha=1.0)

fig.suptitle('Error between Sample Mean Geodesic and Geodesic for Expected Metric')
ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\mathbb{E}\left[\gamma^{1}(t)\right]-\gamma^{1}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\mathbb{E}\left[\gamma^{2}(t)\right]-\gamma^{2}(t)_{\mathbb{E}\left[\mathbf{G}\right]}$')
ax[0].grid()
ax[1].grid()

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')

Z_gamma = vmap(lambda co: vmap(lambda x: param_fun(x, co))(gamma[i]))(coef)
Z_param = vmap(lambda co: vmap(lambda y: vmap(lambda x: param_fun(x,co))(y))(X))(coef)
for i in range(N_sim):
    ax.plot(Z_gamma[i][:,0], Z_gamma[i][:,1], Z_gamma[i][:,2], color='red', alpha=1.0, linewidth=3.0)
    ax.plot_surface(Z_param[i][:,:,0], Z_param[i][:,:,1], Z_param[i][:,:,2], color='cyan', alpha=0.1)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$\gamma$')
ax.set_title('Realisations of stochastic geodesics')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
ax.legend([legend], [r'$\gamma(t)$'], numpoints = 1)