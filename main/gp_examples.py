#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 23:01:38 2022

@author: frederik
"""

#MAYBE USEFUL:
#https://mc-stan.org/docs/2_22/stan-users-guide/reparameterization-section.html#fn40

#%% Modules

import jax.numpy as jnp
from jax import vmap, jacfwd

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib as mpl

import gaussian_proces as gp
import sp
import kernels as km
import rm

#%% Hyper-parameters

sigman = 0.5
n_training = 10
n_test = 100

domain = (-2, 2)

k = km.gaussian_kernel

#%% Noise

coef_training = sp.sim_normal(0.0, sigman**2, n_training)
coef_testing = sp.sim_normal(0.0, sigman**2, n_test)

#%% Data

def basis_fun(x):
    
    return jnp.array([x, x**2])

X_training = jnp.linspace(-1, 1, 10)
y_training = jnp.sum(basis_fun(X_training), axis=0)+coef_training

n_test = 100
X_test = jnp.linspace(-2,2,n_test)
y_test = jnp.sum(basis_fun(X_test), axis=0)+coef_testing

#%% Simulate prior distirbution

GP = gp.gp(X_training, y_training, sigman, k=k)

N_sim = 10
gp_prior = GP.sim_prior(X_test, n_sim=N_sim)
plt.figure(figsize=(8, 6))
for i in range(N_sim):
    plt.plot(X_test, gp_prior[i], linestyle='-', marker='o', markersize=3)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$y = f(x)$', fontsize=13)
plt.title((
    'Realisations from prior distribution'))
plt.xlim(domain)
plt.show()


#%% Simualte posterior distribution

GP = gp.gp(X_training, y_training, sigman, k=k, optimize=True, theta_init = jnp.array([1.0, 1.0]), delta_stable=1e-10)

N_sim = 5
mu_post, cov_post = GP.post_mom(X_test)

gp_post = GP.sim_post(X_test, n_sim=N_sim)

beta = jnp.linspace(0,2,100)
omega = jnp.linspace(0,2,100)
beta, omega = jnp.meshgrid(beta, omega)

pYX = vmap(lambda x1,x2: vmap(lambda y1,y2: GP.log_ml(jnp.array([y1, y2])))(x1,x2))(beta, omega)
theta_opt = GP.opt(jnp.array([1.0, 1.0]))
popt = GP.log_ml(theta_opt)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(beta, omega, pYX, 100)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\omega$')
ax.set_title('$\log p(y|X)$')
ax.scatter(theta_opt[0],theta_opt[1], s=100)
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(beta, omega, pYX, color='red', alpha=0.2)
ax.scatter(theta_opt[0],theta_opt[1], popt, s=100)
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\omega$')
ax.set_zlabel(r'$\log p(y|X)$')

std = jnp.sqrt(jnp.diag(cov_post))

fig, ax = plt.subplots(1,1,figsize=(8, 6))
# Plot the distribution of the function (mean, covariance)
ax.plot(X_test, y_test, 'b--', label='$Observations$')
ax.fill_between(X_test, mu_post-2*std, mu_post+2*std, color='red', 
                 alpha=0.15, label='2 standard deviations')
ax.plot(X_test, mu_post, 'r-', lw=2, label='Predictive mean')
ax.plot(X_training, y_training, 'ko', linewidth=2, label='Training data')
ax.plot(X_test, gp_post.T)
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$y$', fontsize=13)
ax.set_title('Distribution of posterior and prior data.')
#ax.axis([domain[0], domain[1], -3, 3])
ax.legend()

#%% Simulate plane data

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

N_training = 10
sigman = 0.0

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 1.5, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
y_plot = jnp.zeros_like(x1_mesh)

x = jnp.linspace(-2,2,100)
y = jnp.linspace(-2,2,100)
X1, X2 = jnp.meshgrid(x, y)
Z = jnp.zeros_like(X1)

x1 = x1_mesh.reshape(-1)
x2 = x2_mesh.reshape(-1)

X_training = jnp.vstack((x1,x2))

y_training = jnp.vstack((x1,x2,y_plot.reshape(-1)))
RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=10, tol=0.1)

x0 = jnp.array([0.5, 0.5])
x1 = jnp.array([-0.5, -0.5])

gamma, _ = RMEG.geo_ivp(x0, x1)
gamma_manifold, _ = RMEG.post_mom(gamma.T)

RM = rm.rm_geometry(param_fun=lambda x: jnp.array([x[0], x[1], 0.0]), method='runge-kutta')
gamma_true, _ = RM.geo_ivp(x0, x1)

N_sim = 10
#gamma2 = []
#for i in range(N_sim):
#    print(i)
#    val, _ = GP.geo_ivp(x0, x1)
#    print(jnp.max(val))
#    if jnp.max(val)<2:
#        gamma2.append(val)
        
#N_sim = len(gamma2)
#gamma2 = jnp.stack(gamma2)

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

gamma2 = vmap(lambda x: RMSG.geo_ivp(x0, x1, x)[0])(eps)

gamma2_plot = []
for i in range(N_sim):
    if jnp.max(gamma2[i])<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, color='cyan', alpha=0.2)
ax.scatter(x1_mesh, x2_mesh, y_plot, color='black', alpha=0.2)
#ax.plot(gamma[:,0], gamma[:,1], gamma_manifold[2,:], color='black', alpha=1.0)
ax.plot(gamma_manifold[0], gamma_manifold[1], gamma_manifold[2], color='black', alpha=1.0)

for i in range(len(gamma2_plot)):
    ax.plot(gamma_manifold2[i,0], gamma_manifold2[i,1], gamma_manifold2[i,2], color='orange', alpha=0.5)
ax.plot(gamma_true[:,0], gamma_true[:,1], jnp.zeros_like(gamma_true[:,0]), color='red', alpha=1.0)

fig, ax = plt.subplots(1,2,figsize=(8,6))
for i in range(N_sim):
    ax[0].plot(jnp.linspace(0,1,gamma.shape[0]), gamma2_plot[i,:,0], color='cyan', alpha=0.4)
    ax[1].plot(jnp.linspace(0,1,gamma.shape[0]), gamma2_plot[i,:,1], color='cyan', alpha=0.4)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

n_points = 10
x = jnp.linspace(-1.,1.,n_points)
y = jnp.linspace(-1.,1.,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.transpose(jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points))), axes=(1,2,0))
for i in range(N_sim):
    sec = vmap(lambda y1: vmap(lambda y2: RMSG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), eps[i]))(y1))(X)
    #X1_plot
    #X2_plot
    #sec_plot = sec[sec<10]
    ax.plot_surface(X[:,:,0], X[:,:,1], sec, color='cyan', alpha=0.2)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Simulate paraboloid data

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

N_training = 10
sigman = 0.0

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 1.5, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
y_plot = x1_mesh**2+x2_mesh**2

x = jnp.linspace(-1.,1.,100)
y = jnp.linspace(-1.,1.,100)
X1, X2 = jnp.meshgrid(x, y)
Z = X1**2+X2**2

x1 = x1_mesh.reshape(-1)
x2 = x2_mesh.reshape(-1)
y_training = jnp.vstack((x1,x2,y_plot.reshape(-1)))
RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)

test = RMEG.jac_mom(jnp.array([1.0, 1.0]))

x0 = jnp.array([0.5, 0.5])
x1 = jnp.array([-0.5, -0.5])

gamma, _ = RMEG.geo_ivp(x0, x1)
gamma_manifold, _ = RMEG.post_mom(gamma.T)

RM = rm.rm_geometry(param_fun=lambda x: jnp.array([x[0], x[1], x[0]**2+x[1]**2]), method='runge-kutta')
gamma_true, _ = RM.geo_ivp(x0, x1)

N_sim = 10
#gamma2 = []
#for i in range(N_sim):
#    print(i)
#    val, _ = GP.geo_ivp(x0, x1)
#    print(jnp.max(val))
#    if jnp.max(val)<2:
#        gamma2.append(val)
        
#N_sim = len(gamma2)
#gamma2 = jnp.stack(gamma2)

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

gamma2 = vmap(lambda x: RMSG.geo_ivp(x0, x1, x)[0])(eps)

gamma2_plot = []
for i in range(N_sim):
    if jnp.abs(jnp.max(gamma2[i,-1]))<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, color='cyan', alpha=0.2)
ax.scatter(x1_mesh, x2_mesh, y_plot, color='black', alpha=0.2)
#ax.plot(gamma[:,0], gamma[:,1], gamma_manifold[2,:], color='black', alpha=1.0)
ax.plot(gamma_manifold[0], gamma_manifold[1], gamma_manifold[2], color='black', alpha=1.0)

for i in range(len(gamma2_plot)):
    ax.plot(gamma_manifold2[i,0], gamma_manifold2[i,1], gamma_manifold2[i,2], color='orange', alpha=0.5)
ax.plot(gamma_true[:,0], gamma_true[:,1], gamma_true[:,0]**2+gamma_true[:,1]**2, color='red', alpha=1.0)

fig, ax = plt.subplots(1,2,figsize=(8,6))
for i in range(N_sim):
    ax[0].plot(jnp.linspace(0,1,gamma.shape[0]), gamma2_plot[i,:,0], color='cyan', alpha=0.4)
    ax[1].plot(jnp.linspace(0,1,gamma.shape[0]), gamma2_plot[i,:,1], color='cyan', alpha=0.4)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()

plt.figure(figsize=(8,6))
plt.hist(gamma2_plot[:,-1,0], bins=1)

plt.figure(figsize=(8,6))
plt.hist(gamma2_plot[:,-1,1], bins=10)


plt.figure(figsize=(8,6))
plt.hist(gamma_manifold2[:,2], bins=10)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

n_points = 10
x = jnp.linspace(-1.,1.,n_points)
y = jnp.linspace(-1.,1.,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.transpose(jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points))), axes=(1,2,0))
for i in range(N_sim):
    sec = vmap(lambda y1: vmap(lambda y2: RMSG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), eps[i]))(y1))(X)
    #X1_plot
    #X2_plot
    #sec_plot = sec[sec<10]
    ax.plot_surface(X[:,:,0], X[:,:,1], sec, color='cyan', alpha=0.2)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Simulate circle data

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

sigman = 0.0
N_training = 10

import pandas as pd

data_path = 'sim_data/circle.csv'
df = pd.read_csv(data_path, index_col=0)
X = jnp.array(df)
X_training = X[0:-1]
y_training = X

import numpy as np

sigman = 0.0
theta = jnp.linspace(0, 2*np.pi, 100)
x1 = np.cos(theta)
x2 = np.sin(theta)
x3 = np.zeros_like(theta)
df = np.vstack((x1, x2, x3))    
X = jnp.array(df)
X_training = X[0:-1]
y_training = X



RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=10, tol=0.1, method='euler')
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10, max_iter=1, tol=0.1, method='euler')


def circle_fun(theta, mu = jnp.array([1.,1.,1.]), r=1):
    
    x1 = r*jnp.cos(theta)
    x2 = r*jnp.sin(theta)
    x3 = 0
    
    return jnp.array([x1, x2, x3])

x0 = circle_fun(jnp.pi/3)[:-1]
x1 = circle_fun(0.0)[:-1]

gamma, _ = RMEG.geo_bvp(x0, x1)
gamma_manifold, _ = RMEG.post_mom(gamma.T)

N_sim = 10
#gamma2 = []
#for i in range(N_sim):
#    print(i)
#    val, _ = GP.geo_ivp(x0, x1)
#    print(jnp.max(val))
#    if jnp.max(val)<2:
#        gamma2.append(val)
        
#N_sim = len(gamma2)
#gamma2 = jnp.stack(gamma2)

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3).reshape(N_sim, 2, 3)

gamma2 = RMSG.geo_bvp(x0,x1,eps[0])

gamma2 = vmap(lambda x: RMSG.geo_ivp(x0, x1, x)[0])(eps)

gamma2_plot = []
for i in range(N_sim):
    if jnp.abs(jnp.max(gamma2[i,-1]))<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_training[0], X_training[1], y_training[2], color='black', alpha=0.2)
#ax.plot(gamma[:,0], gamma[:,1], gamma_manifold[2,:], color='black', alpha=1.0)
ax.plot(gamma_manifold[0], gamma_manifold[1], gamma_manifold[2], color='black', alpha=1.0)

for i in range(len(gamma2_plot)):
    ax.plot(gamma_manifold2[i,0], gamma_manifold2[i,1], gamma_manifold2[i,2], color='orange', alpha=0.5)

fig, ax = plt.subplots(1,2,figsize=(8,6))
for i in range(N_sim):
    ax[0].plot(jnp.linspace(0,1,gamma.shape[0]), gamma2_plot[i,:,0], color='cyan', alpha=0.4)
    ax[1].plot(jnp.linspace(0,1,gamma.shape[0]), gamma2_plot[i,:,1], color='cyan', alpha=0.4)

ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()

plt.figure(figsize=(8,6))
plt.hist(gamma2_plot[:,-1,0], bins=1)

plt.figure(figsize=(8,6))
plt.hist(gamma2_plot[:,-1,1], bins=10)


plt.figure(figsize=(8,6))
plt.hist(gamma_manifold2[:,2], bins=10)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

n_points = 10
x = jnp.linspace(-1.,1.,n_points)
y = jnp.linspace(-1.,1.,n_points)
X1, X2 = jnp.meshgrid(x, y)
X = jnp.transpose(jnp.concatenate((X1.reshape(1,n_points, n_points), X2.reshape(1,n_points, n_points))), axes=(1,2,0))
for i in range(N_sim):
    sec = vmap(lambda y1: vmap(lambda y2: RMSG.SC(y2, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), eps[i]))(y1))(X)
    #X1_plot
    #X2_plot
    #sec_plot = sec[sec<10]
    ax.plot_surface(X[:,:,0], X[:,:,1], sec, color='cyan', alpha=0.2)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend], ['Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()

