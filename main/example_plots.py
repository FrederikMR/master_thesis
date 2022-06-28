#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:16:50 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib as mpl

import rm_param as pm

#%% Hyper-parameters

n_points = 100

#%% Plots - Plane

grid0x1 = -2
grid1x1 = 2
grid0x2 = -2
grid1x2 = 2
n_coef = 2
y0 = jnp.array([1.0,1.0])
yT = jnp.array([-1.0,1.0])
param_fun = pm.plane

x = jnp.linspace(grid0x1,grid1x1,n_points)
y = jnp.linspace(grid0x2,grid1x2,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
X = jnp.einsum('ijk->jki', X)

Z = vmap(lambda y: vmap(lambda x: param_fun(x,jnp.array([1.0, 1.0])))(y))(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Z[:,:,0], Z[:,:,1], Z[:,:,2], color='red', alpha=1.0)
ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$x^{1}+x^{2}$')

plt.tight_layout()
plt.show()

#%% Plots - Sphere

epsilon = 1e-2 #For numerical stability in the boundary
grid0x1 = 0.0+epsilon
grid1x1 = 2*jnp.pi-epsilon
grid0x2 = 0.0+epsilon
grid1x2 = jnp.pi-epsilon
y0 = jnp.array([0.5*jnp.pi,0.5*jnp.pi])
yT = jnp.array([jnp.pi,0.75*jnp.pi])
param_fun = pm.sphere

x = jnp.linspace(grid0x1,grid1x1,n_points)
y = jnp.linspace(grid0x2,grid1x2,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
X = jnp.einsum('ijk->jki', X)

Z = vmap(lambda y: vmap(lambda x: param_fun(x,jnp.array([1.0])))(y))(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Z[:,:,0], Z[:,:,1], Z[:,:,2], color='red', alpha=1.0)
ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'Unit Sphere')

plt.tight_layout()
plt.show()

#%% Plots - Paraboloid

grid0x1 = -2
grid1x1 = 2
grid0x2 = -2
grid1x2 = 2
y0 = jnp.array([1.0,1.0])
yT = jnp.array([-1.0,1.0])
N_degree = 2
n_coef = N_degree
param_fun = lambda x, coef: pm.polynomial_1d(x, coef, N_degree)

x = jnp.linspace(grid0x1,grid1x1,n_points)
y = jnp.linspace(grid0x2,grid1x2,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points)))
X = jnp.einsum('ijk->jki', X)

Z = vmap(lambda y: vmap(lambda x: param_fun(x,jnp.array([1.0, 1.0])))(y))(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Z[:,:,0], Z[:,:,1], Z[:,:,2], color='red', alpha=1.0)
ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$\left(x^{1}\right)^{2}+\left(x^{2}\right)^{2}$')

plt.tight_layout()
plt.show()