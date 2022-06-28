#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:21:58 2022

@author: frederik
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap, jit
#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

#%% Bayes class

class bayes(object):
    
    def __init__(self):
        
        self.post_w = None
        self.post_f = None

#%% Functions

def bayes_reg(X_training, y_training, sigman=1.0, Sigmap = None, basis_fun=None):
    
    def post_weights():
        
        A = Phi.dot(Phi.T)/sigman2+Sigmap_inv
        A_inv = jnp.linalg.inv(A)
        
        mu_post = A_inv.dot(Phi.dot(y_training))/sigman2
        cov_post = A_inv
        
        return mu_post, cov_post
    
    def post_f(X_test):
        
        if X_test.ndim==1:
            phi_test = X_test
        else:
            phi_test = vmap(basis_fun)(X_test.T).T
        
        if phi_test.ndim == 1:
            phi_test = phi_test.reshape(1,-1)
        
        SigmaPhi = Sigmap.dot(Phi)
        
        K = Phi.T.dot(Sigmap).dot(Phi)+jnp.eye(N_training)*sigman2
        solved = jnp.linalg.solve(K.T, SigmaPhi.T).T
        
        mu_post = phi_test.T.dot(solved).dot(y_training)
        cov_post = phi_test.T.dot(Sigmap).dot(phi_test)-phi_test.T.dot(solved).dot(SigmaPhi.T).dot(phi_test)
        
        return mu_post, cov_post
    
    if basis_fun is None:
        Phi = X_training
    else:
        Phi = vmap(basis_fun)(X_training.T).T
        
    if Phi.ndim == 1:
        Phi = Phi.reshape(1,-1)

    n_dim, N_training = Phi.shape
    
    if Sigmap is None:
        Sigmap = jnp.eye(n_dim)
        
    sigman2 = sigman**2
    Sigmap_inv = jnp.linalg.inv(Sigmap)
    
    BR = bayes()
    
    BR.post_w = jit(post_weights)
    BR.post_f = jit(post_f)
    
    return BR