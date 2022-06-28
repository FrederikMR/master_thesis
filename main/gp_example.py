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

import gp
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
    
    x_diff = x-y
    
    return -omega*x_diff*k_fun(x,y,beta,omega)

def DDk_fun(x,y, beta=1.0, omega=1.0):
    
    N = len(x)
    anti_I = jnp.flipud(jnp.eye(N))
    x_diff = (x-y).reshape(1,-1)
    
    return omega*(x_diff.T.dot(x_diff)*omega*jnp.eye(N)-anti_I)*k(x,y,beta,omega)

N_training = 10
sigman = 0.5
coef_training = sp.sim_normal(0.0, sigman**2, N_training*N_training).reshape(N_training, N_training)

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 1.5, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
y_plot = jnp.zeros_like(x1_mesh)+coef_training

x = jnp.linspace(-2,2,100)
y = jnp.linspace(-2,2,100)
X1, X2 = jnp.meshgrid(x, y)
Z = jnp.zeros_like(X1)

x1 = x1_mesh.reshape(-1)
x2 = x2_mesh.reshape(-1)
y_training = y_plot.reshape(-1)

X_training = jnp.vstack((x1,x2))

GP = gp.gp(X_training, y_training, sigman=sigman, k=k_fun, Dk_fun = Dk_fun, DDk_fun = None, delta_stable=1e-10)

test = GP.jac_mom(jnp.array([1.0, 1.0]))

RM = rm.rm_geometry(G=GP.Emmf, method='runge-kutta')

x0 = jnp.array([0.5, 0.5])
x1 = jnp.array([-0.5, -0.5])

gamma, _ = GP.geo_Eivp(x0, x1)
gamma_manifold, _ = GP.post_mom(gamma.T)

RM = rm.rm_geometry(param_fun=lambda x: jnp.array([x[0], x[1], 0.0]), method='runge-kutta')
gamma_true, _ = RM.geo_ivp(x0, x1)

N_sim = 100
#gamma2 = []
#for i in range(N_sim):
#    print(i)
#    val, _ = GP.geo_ivp(x0, x1)
#    print(jnp.max(val))
#    if jnp.max(val)<2:
#        gamma2.append(val)
        
#N_sim = len(gamma2)
#gamma2 = jnp.stack(gamma2)

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim).reshape(N_sim, 2)
gamma2 = vmap(lambda x: GP.geo_ivp(x0, x1, x)[0])(eps)

N_sim = 100
gamma2_plot = []
for i in range(N_sim):
    if jnp.max(gamma2[i])<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(GP.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, color='cyan', alpha=0.2)
ax.scatter(x1_mesh, x2_mesh, y_plot, color='black', alpha=1.0)
ax.plot(gamma[:,0], gamma[:,1], gamma_manifold, color='black', alpha=1.0)
for i in range(len(gamma2_plot)):
    ax.plot(gamma2_plot[i,:,0], gamma2_plot[i,:,0], gamma_manifold2[i], color='orange', alpha=1.0)
ax.plot(gamma_true[:,0], gamma_true[:,1], jnp.zeros_like(gamma_true[:,0]), color='red', alpha=1.0)

#%% Simulate paraboloid data

def k_fun(x,y, beta=1.0, omega=1.0):
    
    x_diff = y-x
    
    return beta*jnp.exp(-omega*jnp.dot(x_diff, x_diff)/2)

def Dk_fun(x,y, beta=1.0, omega=1.0):
    
    x_diff = y-x
    
    return omega*x_diff*k_fun(x,y,beta,omega)


#DDK_fun IS MOSTLY LIKELY WRONG
def DDk_fun(x,y, beta=1.0, omega=1.0):
    
    N = len(x)
    anti_I = jnp.flipud(jnp.eye(N))
    x_diff = (x-y).reshape(1,-1)
    
    return omega*(x_diff.T.dot(x_diff)*omega*jnp.eye(N)-anti_I)*k(x,y,beta,omega)

N_training = 10
coef_training = sp.sim_normal(0.0, sigman**2, N_training*N_training).reshape(N_training, N_training)

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 2, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
y_plot = x1_mesh**2+x2_mesh**2+coef_training

x = jnp.linspace(-1.,1.,100)
y = jnp.linspace(-1.,1.,100)
X1, X2 = jnp.meshgrid(x, y)
Z = X1**2+X2**2

x1 = x1_mesh.reshape(-1)
x2 = x2_mesh.reshape(-1)
y_training = y_plot.reshape(-1)

X_training = jnp.vstack((x1,x2))

GP = gp.gp(X_training, y_training, sigman=sigman, k=k_fun, Dk_fun = Dk_fun, DDk_fun = None)
RM = rm.rm_geometry(G=GP.Emmf, method='euler')

x0 = jnp.array([0.5, 0.5])
x1 = jnp.array([-0.5, -0.5])

x0 = jnp.array([1.0, 1.0])
#x1 = jnp.array([1.0, -1.0])

RM.chris(x1)

gamma, _ = GP.geo_Eivp(x0, x1)
gamma_manifold, _ = GP.post_mom(gamma.T)


RM = rm.rm_geometry(param_fun=lambda x: jnp.array([x[0], x[1], x[0]**2+x[1]**2]), method='euler')
gamma_true, _ = RM.geo_ivp(x0, x1)
gammaz_true = jnp.array([gamma_true[:,0], gamma_true[:,1], gamma_true[:,0]**2+gamma_true[:,1]**2])

#N_sim = 100
#gamma2 = []
#for i in range(N_sim):
#    print(i)
#    val, _ = GP.geo_ivp(x0, x1)
#    print(jnp.max(val))
#    if jnp.max(val)<2:
#        gamma2.append(val)
        
eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim).reshape(N_sim, 2)
gamma2 = vmap(lambda x: GP.geo_ivp(x0, x1, x)[0])(eps)

N_sim = 100
gamma2_plot = []
for i in range(N_sim):
    if jnp.max(gamma2[i])<2:
        gamma2_plot.append(gamma2[i])
gamma2_plot = jnp.stack(gamma2_plot)
        
gamma_manifold2, _ = vmap(GP.post_mom)(jnp.einsum('ijk->ikj', gamma2_plot))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, color='cyan', alpha=0.2)
ax.scatter(x1_mesh, x2_mesh, y_plot, color='black', alpha=1.0)
ax.plot(gamma[:,0], gamma[:,1], gamma_manifold, color='black', alpha=1.0)
for i in range(len(gamma2_plot)):
    ax.plot(gamma2_plot[i,:,0], gamma2_plot[i,:,0], gamma_manifold2[i], color='orange', alpha=1.0)
ax.plot(gamma_true[:,0], gamma_true[:,1], gamma_true[:,0]**2+gamma_true[:,1]**2, color='red', alpha=1.0)


plt.figure(figsize=(8,6))
plt.hist(gamma2_plot[:,-1,0], bins=1)

plt.figure(figsize=(8,6))
plt.hist(gamma2_plot[:,-1,1], bins=10)


plt.figure(figsize=(8,6))
plt.hist(gamma_manifold2, bins=10)


#%% Simulate sphere data

def sphere(x, R=1):
    
    theta = x[0]
    phi = x[1]
    
    return R*jnp.array([jnp.cos(theta)*jnp.sin(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(phi)])

N_training = 20
sigman = 0.1
coef_training = sp.sim_normal(0.0, sigman**2, N_training*N_training).reshape(N_training, N_training)

x1 = jnp.linspace(0,2*jnp.pi,N_training)
x2 = jnp.linspace(0,jnp.pi,N_training)
theta1, phi1 = jnp.meshgrid(x1, x2)
x1_mesh = jnp.cos(theta1)*jnp.sin(phi1)
x2_mesh = jnp.sin(theta1)*jnp.sin(phi1)
y_plot = jnp.cos(phi1)+coef_training

x = jnp.linspace(0,2*jnp.pi,100)
y = jnp.linspace(0,jnp.pi,100)
theta, phi = jnp.meshgrid(x, y)
X1 = jnp.cos(theta)*jnp.sin(phi)
X2 = jnp.sin(theta)*jnp.sin(phi)
Z = jnp.cos(phi)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, color='cyan', alpha=0.2)
ax.scatter(x1_mesh, x2_mesh, y_plot, color='black', alpha=1.0)

x1 = (jnp.cos(x1_mesh)*jnp.sin(x2_mesh)).reshape(-1)
x2 = (jnp.sin(x1_mesh)*jnp.sin(x2_mesh)).reshape(-1)
y_training = y_plot.reshape(-1)

X_training = jnp.vstack((x1,x2))

GP = gp.gp(X_training, y_training, sigman=sigman, k=k, Dk_fun = None, DDk_fun = None, optimize=False, theta_init = jnp.array([1.0, 1.0]))

RM = rm.rm_geometry(G=GP.Emmf)

theta0, phi0 = 0.5*jnp.pi, 0.5*jnp.pi
theta1, phi1 = jnp.pi, 0.75*jnp.pi

x0 = jnp.array([jnp.cos(theta0)*jnp.sin(phi0), jnp.sin(theta0)*jnp.sin(phi0)])
x1 = jnp.array([jnp.cos(theta1)*jnp.sin(phi1), jnp.sin(theta1)*jnp.sin(phi1)])

gamma, _ = GP.geo_Ebvp(x0, x1)
gamma_manifold, _ = GP.post_mom(gamma.T)

RM = rm.rm_geometry(param_fun=sphere)
gamma_true, _ = RM.geo_ivp(jnp.array([theta0, phi0]), jnp.array([theta1, phi1]))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, color='cyan', alpha=0.2)
ax.plot(gamma[:,0], gamma[:,1], gamma_manifold, color='black', alpha=1.0)
ax.plot(jnp.cos(gamma_true[:,0])*jnp.sin(gamma_true[:,1]), jnp.sin(gamma_true[:,0])*jnp.sin(gamma_true[:,1]), gamma_true[:,1], jnp.cos(gamma_true[:,1]), color='red', alpha=1.0)



























