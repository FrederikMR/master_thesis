#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:23:21 2022

@author: frederik
"""
#INSPIRATION:
#https://math.stackexchange.com/questions/1834355/matrix-by-matrix-derivative-formula?rq=1


#%% Modules

import jax.numpy as jnp
from jax import jacfwd

import kernels as km
import sp

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

#%% Kernel matrix and cholesky

K = km.km(X_training, kernel_fun=k)

L = jnp.linalg.cholesky(K)

#%% fun

def fun(X):
    
    K = km.km(X, kernel_fun=k)

    L = jnp.linalg.cholesky(K)
    
    return L


fun_grad = jacfwd(fun)
test = fun_grad(X_training)