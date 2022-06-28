#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:08:30 2022

@author: frederik
"""

#%% Sources

#http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf

#%% Modules

import jax.numpy as jnp
from jax import grad, jacrev, jacfwd, vmap, jit, lax
#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

from scipy.optimize import minimize

import kernels as km
import sp
import ode_integrator as oi

#%% GP class

class gp_features(object):
    
    def __init__(self):
        
        self.G = None
        self.J = None

#%% Functions

################################### GEODESIC EQUATION IS WRONG - COMPUTE IT THROUGH CHRISTOFFEL SYMBOLS
def gp(X_training, y_training, sigman = 1.0, m_fun = None, Dm_fun = None, k = None, Dk_fun = None, DDk_fun = None,
      theta_init = None, optimize=False, grad_K = None,  max_iter = 100, delta_stable=1e-10):
    
    def sim_prior(X, n_sim=10):
        
        if X.ndim == 1:
            X = X.reshape(1,-1)
        
        _, N_data = X.shape
        
        if m_fun is None:
            mu = jnp.zeros(N_data)
        else:
            mu = m_fun(X)
        
        K = km.km(X.T, kernel_fun = k_fun)+jnp.eye(N_data)*delta_stable
        #gp = jnp.asarray(np.random.multivariate_normal(mean = mu,
        #                                              cov = K,
        #                                              size=n_sim))
        
        gp = sp.sim_multinormal(mu=mu, cov=K, dim=n_sim)
        
        return gp
    
    def post_mom(X_test):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)

        m_test = m_fun(X_test)
        
        K21 = km.km( X_training.T, X_test.T, k_fun)
        K22 = km.km(X_test.T, X_test.T, k_fun)
        
        solved = jnp.linalg.solve(K11, K21).T
        
        mu_post = m_test+(solved @ (y_training-m_training))
        cov_post = K22-(solved @ K21)
        
        return mu_post, cov_post
    
    def sim_post(X_test, n_sim=10):
        
        mu_post, cov_post = GP.post_mom(X_test)
        #gp = jnp.asarray(np.random.multivariate_normal(mean = mu_post,
        #                                              cov = cov_post,
        #                                              size=n_sim))
        gp = sp.sim_multinormal(mu=mu_post, cov=cov_post, dim=n_sim)
        
        return gp
    
    def log_ml(theta):

        K11 = K11_theta(theta)
     
        pYX = -0.5*(y_training.dot(jnp.linalg.solve(K11, y_training))+jnp.log(jnp.linalg.det(K11))+N_training*jnp.log(2*jnp.pi))
        
        return pYX
    
    def optimize_hyper(theta_init):

        #def grad_pYX(theta):
        #    
        #    K11 = K11_theta(theta)+jnp.eye(N_training)*delta_stable
        #    K11_inv = jnp.linalg.inv(K11)
        #    K_theta = grad_K(theta)
        #    
        #    alpha = jnp.linalg.solve(K11, y_training).reshape(-1,1)
        #    alpha_mat = alpha.dot(alpha.T)
        #                            
        #    return 0.5*jnp.trace((alpha_mat-K11_inv).dot(K_theta))

        #sol = minimize(lambda theta: -log_ml(theta), theta_init, jac=grad_pYX,
        #             method='BFGS', options={'gtol': tol, 'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False
        
        sol = minimize(lambda theta: -log_ml(theta), theta_init,
                     method='Nelder-Mead', options={'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    def jacobian_mom(X_test):

        X_test = X_test.reshape(-1)        
        Dm_test = Dm_fun(X_test.reshape(1,-1))    
        
        #DK = vmap(lambda x: Dk_fun(X_training.T, x))(X_test.T)
        DK = vmap(lambda x: Dk_fun(x, X_test))(X_training.T)
        DDK = DDk_fun(X_test, X_test)  
    
        solved = jnp.linalg.solve(K11, DK).T
        
        mu_post = Dm_test+(solved @ (y_training-Dm_training))
        
        cov_post = DDK-(solved @ DK)
        
        return mu_post, -cov_post #SOMETHING WRONG COV IS NOT POSTIVE (SYMMETRIC DEFINITE), BUT IS NEGATIVE SYMMETRIC DEFINITE
    
    def sim_jacobian(X_test):
        
        mu_post, cov_post = jacobian_mom(X_test)
        
        #J = jnp.asarray(np.random.multivariate_normal(mean = mu_post,
        #                                              cov = cov_post))
        
        J = sp.sim_multinormal(mu=mu_post, cov=cov_post)
        
        return J
    
    def Emmf(X_test):
        
        p = 1 #p = y_training.shape[-1]
        mu_post, cov_post = GP.jac_mom(X_test)
        
        mu_post = mu_post.reshape(1,-1)
        
        EG = mu_post.T.dot(mu_post)+p*cov_post
        
        return EG
    
    def DEJ(X_test):
        
        X_test = X_test.reshape(-1)
        
        DDK12 = jnp.einsum('ijk->jki', vmap(lambda x: DDk_fun(X_test, x))(X_training.T))
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T).T
        
        K11_inv = jnp.linalg.inv(K11)
        
        solved1 = DDK12.dot(K11_inv)
        solved2 = DK12.dot(K11_inv)
        
        #MISSING M(X), DM(X), DDM(X)
        
        return solved1.dot(y_training)-solved2.dot(DK11.dot(K11_inv).dot(y_training))
    
    def Dcov(X_test):
        
        X_test = X_test.reshape(-1)
        
        DDK12 = jnp.einsum('ijk->jki', vmap(lambda x: DDk_fun(X_test, x))(X_training.T))
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T).T
        DDDK = DDDk_fun(X_test, X_test)
        
        
        K11_inv = jnp.linalg.inv(K11)
        
        solved1 = DDK12.dot(K11_inv)
        solved2 = DK12.dot(K11_inv)

        return DDDK-solved1.dot(DK12.T)+solved2.dot(jnp.einsum('jki->ijk', DK11.dot(K11_inv).dot(DK12.T))-jnp.einsum('ijk->ikj', DDK12))
    
    def DEmmf(X_test):
        
        J, _ = GP.jac_mom(X_test)
        J = J.reshape(1,-1)
        DJ = GP.DEJ(X_test).reshape(1,dim,dim)
        DJ2 = jnp.einsum('ijk->jki', DJ)
        cov = GP.Dcov(X_test)
        
        return jnp.einsum('jki,im->jkm', DJ2, J)+jnp.einsum('ji,ikm->jkm', J.T, DJ)+cov
    
    #@partial(jit, static_argnums=(2,))
    def Dmmf(X_test, epsilon):
        
        def cov_fun(x):
            
            _, cov = GP.jac_mom(x)
            
            L = jnp.linalg.cholesky(cov)
            
            return L
        
        
        J, cov = GP.jac_mom(X_test)
        J = J.reshape(1,-1)
        DJ = GP.DEJ(X_test).reshape(1,dim,dim)
        DJ2 = jnp.einsum('ijk->jki', DJ)
        
        W = epsilon.dot(epsilon.T)

        L = jnp.linalg.cholesky(cov)
        
        
        DL = jacfwd(cov_fun)(X_test)
        
        termJ = jnp.einsum('jki,im->jkm', DJ2, J)+jnp.einsum('ji,ikm->jkm', J.T, DJ)
        
        term1 = jnp.einsum('jik,im->jmk', DL, W)
        term2 = jnp.einsum('jmk,im->jik', term1, L)
        term3 = jnp.einsum('ji,im->jm', L, W)
        term4 = jnp.einsum('jm,imk->jik', term3, DL)
        term11 = term2+term4
        
        term5 = jnp.einsum('ji,im->jm', epsilon, J)
        term6 = jnp.einsum('jik,im->jmk', DL, term5)
        term7 = jnp.einsum('mi,ijk->mjk', epsilon, DJ)
        term8 = jnp.einsum('im,mjk->ijk', L, term7)
        term22 = term6+term8
        
        term33 = jnp.einsum('ijk->jik', term8)
        
        G = (J+L.dot(epsilon.reshape(-1))).reshape(-1,1)
        G = G.dot(G.T)
        
        return G, termJ+term11+term22+term33
    
    def Ebvp_geodesic(x,y, grid=jnp.linspace(0,1,100), tol=1e-05, method='runge-kutta'):
        
        def chris_symbols(x):
            
            G = GP.Emmf(x)
            
            G_inv = jnp.linalg.inv(G)
            DG = GP.DEmmf(x)
            
            chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                         +jnp.einsum('lij, lm->mji', DG, G_inv) \
                         -jnp.einsum('ijl, lm->mji', DG, G_inv))
            
            return chris
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = chris_symbols(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def Eivp_geodesic(x,v, grid=jnp.linspace(0,1,100), tol=1e-05, method='runge-kutta'):
        
        def chris_symbols(x):
            
            G = GP.Emmf(x)
            
            G_inv = jnp.linalg.inv(G)
            DG = GP.DEmmf(x)
            
            chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                         +jnp.einsum('lij, lm->mji', DG, G_inv) \
                         -jnp.einsum('ijl, lm->mji', DG, G_inv))
            
            return chris
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = chris_symbols(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)        
        
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def ivp_geodesic(x,v, eps = None, grid=jnp.linspace(0,1,1000), tol=1e-05, method='runge-kutta'):
        
        def eq_geodesic(t, y, eps):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            G, DG = GP.Dmmf(gamma, eps)
            G = G+jnp.eye(dim)*1e-2
            
            DG = jnp.einsum('ijk->kij', DG).reshape(dim,-1, order='F')
                        
            return jnp.concatenate((Dgamma, -1/2*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma, Dgamma))))
                    
        def step_fun(yn, time):
        
            #tn, hn = time
            #hn2 = hn/2
            
            #k1 = eq_geodesic(tn, yn, eps)
            #k2 = eq_geodesic(tn+hn2, yn+hn2*k1, eps)
            #k3 = eq_geodesic(tn+hn2, yn+hn2*k2, eps)
            #k4 = eq_geodesic(tn+hn, yn+hn*k3, eps)
            
            #y = yn+hn*(k1+2*k2+2*k3+k4)
            
            #return y, y
        
            tn, hn = time
            y = yn+eq_geodesic(tn,yn, eps)*hn
            
            return y, y
        
        N = len(x)
        
        dt_grid = jnp.hstack((jnp.diff(grid)))
        x0 = jnp.concatenate((x, v), axis=0)
        if eps is None:
            eps = sp.sim_multinormal(mu=jnp.zeros(dim), cov=jnp.eye(dim), dim=1).reshape(dim,1)
        else:
            eps = eps.reshape(dim,1)
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        yt = jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
        
        gamma = yt[:,0:len(x)]
        Dgamma = yt[:,len(x):]
        
        return gamma, Dgamma

    def mmf(X_test):
        
        J = GP.J(X_test).reshape(1,-1)
        
        return J.T.dot(J)
    
    sigman2 = sigman**2
    
    if X_training.ndim == 1:
        X_training = X_training.reshape(1,-1)
    
    dim, N_training = X_training.shape
    
    if Dm_fun is None and m_fun is not None:
        Dm_fun = vmap(lambda x: grad(m_fun))
    elif m_fun is None and Dm_fun is None:
        Dm_fun = lambda x: jnp.zeros(x.shape[-1])
    
    if m_fun is None:
        m_fun = lambda x: jnp.zeros(x.shape[-1])
        
    if k is None:
        k = km.gaussian_kernel
        
    K11_theta = lambda theta: km.km(X_training, X_training, lambda x,y: k(x,y,*theta))+sigman2*jnp.eye(N_training)
    if grad_K is None:
        grad_K = jacfwd(K11_theta)
    
    if optimize:
        theta = optimize_hyper(theta_init)
        k_fun = lambda x,y: k(x,y,*theta)
    else:
        k_fun = k
    
    if Dk_fun is None:
        Dk_fun = grad(k_fun, argnums=0)
    if DDk_fun is None:
        #DDk_fun = jacfwd(grad(k_fun, argnums=0), argnums=0)
        DDk_fun = jacfwd(jacrev(k_fun, argnums=0), argnums=0)
        
    DDDk_fun = jacfwd(DDk_fun)
    
    K11 = km.km(X_training.T, X_training.T, k_fun)+sigman2*jnp.eye(N_training)+jnp.eye(N_training)*delta_stable
    
    DK11 = jnp.einsum('ijk->jki', vmap(lambda y: vmap(lambda x: Dk_fun(x, y))(X_training.T))(X_training.T))
    
    m_training = m_fun(X_training)
    Dm_training = Dm_fun(X_training)
    
    GP = gp_features() 
    GP.sim_prior = sim_prior
    GP.post_mom = jit(post_mom)
    GP.sim_post = sim_post
    GP.log_ml = jit(log_ml)
    GP.opt = optimize_hyper
    GP.jac_mom = jacobian_mom
    GP.J = sim_jacobian
    GP.Emmf = jit(Emmf)
    GP.mmf = mmf
    GP.DEJ = jit(DEJ)
    GP.Dcov = jit(Dcov)
    GP.DEmmf = jit(DEmmf)
    GP.Dmmf = Dmmf
    GP.geo_Ebvp = Ebvp_geodesic
    GP.geo_Eivp = Eivp_geodesic
    GP.geo_ivp = ivp_geodesic
    
    return GP

#%% Functions

################################### GEODESIC EQUATION IS WRONG - COMPUTE IT THROUGH CHRISTOFFEL SYMBOLS
def gp2(X_training, y_training, sigman = 1.0, m_fun = None, Dm_fun = None, k = None, Dk_fun = None, DDk_fun = None,
      theta_init = None, optimize=False, grad_K = None,  max_iter = 100, delta_stable=1e-10):
    
    def sim_prior(X, n_sim=10):
        
        if X.ndim == 1:
            X = X.reshape(1,-1)
        
        _, N_data = X.shape
        
        if m_fun is None:
            mu = jnp.zeros(N_data)
        else:
            mu = m_fun(X)
        
        K = km.km(X.T, kernel_fun = k_fun)+jnp.eye(N_data)*delta_stable
        #gp = jnp.asarray(np.random.multivariate_normal(mean = mu,
        #                                              cov = K,
        #                                              size=n_sim))
        
        return sp.sim_multinormal(mu=mu, cov=K, dim=n_sim)
    
    def post_mom(X_test):
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)

        m_test = m_fun(X_test)
        
        K21 = km.km(X_training.T, X_test.T, k_fun)
        K22 = km.km(X_test.T, X_test.T, k_fun)
        
        solved = jnp.linalg.solve(K11, K21).T
        if N_obs == 1:
            mu_post = m_test+(solved @ (y_training-m_training))
        else:
            mu_post = vmap(lambda y: m_test+(solved @ (y-m_training)))(y_training.T)
            
        cov_post = K22-(solved @ K21)
        
        return mu_post, cov_post
    
    def sim_post(X_test, n_sim=10):
        
        mu_post, cov_post = GP.post_mom(X_test)
        #gp = jnp.asarray(np.random.multivariate_normal(mean = mu_post,
        #                                              cov = cov_post,
        #                                              size=n_sim))
        if N_obs == 1:
            gp = sp.sim_multinormal(mu=mu_post, cov=cov_post, dim=n_sim)
        else:
            gp = vmap(lambda mu: sp.sim_multinormal(mu=mu_post, cov=cov_post, dim=n_sim))(mu_post)
        
        return gp
    
    def log_ml(theta):

        K11 = K11_theta(theta)
        
        if N_obs == 1:
            pYX = -0.5*(y_training.dot(jnp.linalg.solve(K11, y_training))+jnp.log(jnp.linalg.det(K11))+N_training*jnp.log(2*jnp.pi))
        else:
            pYX = vmap(lambda y: (y_training.dot(jnp.linalg.solve(K11, y))+jnp.log(jnp.linalg.det(K11))+N_training*jnp.log(2*jnp.pi)))(y_training.T)
            pYX = -0.5*jnp.sum(pYX)
             
        return pYX
    
    def optimize_hyper(theta_init):

        #def grad_pYX(theta):
        #    
        #    K11 = K11_theta(theta)+jnp.eye(N_training)*delta_stable
        #    K11_inv = jnp.linalg.inv(K11)
        #    K_theta = grad_K(theta)
        #    
        #    alpha = jnp.linalg.solve(K11, y_training).reshape(-1,1)
        #    alpha_mat = alpha.dot(alpha.T)
        #                            
        #    return 0.5*jnp.trace((alpha_mat-K11_inv).dot(K_theta))

        #sol = minimize(lambda theta: -log_ml(theta), theta_init, jac=grad_pYX,
        #             method='BFGS', options={'gtol': tol, 'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False
        
        sol = minimize(lambda theta: -log_ml(theta), theta_init,
                     method='Nelder-Mead', options={'maxiter':max_iter, 'disp':True}) #FOR NO MESSAGE: SET 'disp':False

        return sol.x
    
    def jacobian_mom(X_test):

        X_test = X_test.reshape(-1)        
        Dm_test = Dm_fun(X_test.reshape(1,-1))    
        
        #DK = vmap(lambda x: Dk_fun(X_training.T, x))(X_test.T)
        DK = vmap(lambda x: Dk_fun(x, X_test))(X_training.T)
        DDK = DDk_fun(X_test, X_test)  
    
        solved = jnp.linalg.solve(K11, DK).T
        
        if N_obs == 1:
            mu_post = Dm_test+(solved @ (y_training-Dm_training))
        else:
            mu_post = vmap(lambda y: Dm_test+(solved @ (y-Dm_training)))(y_training.T)
        
        cov_post = DDK-(solved @ DK)
        
        return mu_post, cov_post #SOMETHING WRONG COV IS NOT POSTIVE (SYMMETRIC DEFINITE), BUT IS NEGATIVE SYMMETRIC DEFINITE
    
    def sim_jacobian(X_test):
        
        mu_post, cov_post = jacobian_mom(X_test)
        
        #J = jnp.asarray(np.random.multivariate_normal(mean = mu_post,
        #                                              cov = cov_post))
        
        if N_obs == 1:
            J = sp.sim_multinormal(mu=mu_post, cov=cov_post)
        else:
            J = vmap(lambda mu: sp.sim_multinormal(mu=mu, cov=cov_post))(mu_post)
    
        return J
    
    def Emmf(X_test):

        mu_post, cov_post = GP.jac_mom(X_test)
        
        if N_obs == 1:    
            mu_post = mu_post.reshape(-1,1)
        
        EG = mu_post.dot(mu_post.T)+N_obs*cov_post
        
        return EG
    
    def DEJ(X_test):
        
        X_test = X_test.reshape(-1)
        
        DDK12 = jnp.einsum('ijk->jki', vmap(lambda x: DDk_fun(X_test, x))(X_training.T))
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T).T
        
        K11_inv = jnp.linalg.inv(K11)
        
        solved1 = DDK12.dot(K11_inv)
        solved2 = DK12.dot(K11_inv)
        
        #MISSING M(X), DM(X), DDM(X)
        
        return vmap(lambda y: solved1.dot(y)-solved2.dot(DK11.dot(K11_inv).dot(y)))(y_training.T)
    
    def Dcov(X_test):
        
        X_test = X_test.reshape(-1)
        
        DDK12 = jnp.einsum('ijk->jki', vmap(lambda x: DDk_fun(X_test, x))(X_training.T))
        DK12 = vmap(lambda x: Dk_fun(X_test, x))(X_training.T).T
        DDDK = DDDk_fun(X_test, X_test)
        
        
        K11_inv = jnp.linalg.inv(K11)
        
        solved1 = DDK12.dot(K11_inv)
        solved2 = DK12.dot(K11_inv)

        return DDDK-solved1.dot(DK12.T)+solved2.dot(jnp.einsum('jki->ijk', DK11.dot(K11_inv).dot(DK12.T))-jnp.einsum('ijk->ikj', DDK12))
    
    def DEmmf(X_test):
        
        J, _ = GP.jac_mom(X_test)
        J = J.reshape(1,-1)
        DJ = GP.DEJ(X_test).reshape(1,dim,dim)
        DJ2 = jnp.einsum('ijk->jki', DJ)
        cov = GP.Dcov(X_test)
        
        return jnp.einsum('jki,im->jkm', DJ2, J)+jnp.einsum('ji,ikm->jkm', J.T, DJ)+cov
    
    #@partial(jit, static_argnums=(2,))
    def Dmmf(X_test, epsilon):
        
        def cov_fun(x):
            
            _, cov = GP.jac_mom(x)
            
            L = jnp.linalg.cholesky(cov)
            
            return L
        
        
        J, cov = GP.jac_mom(X_test)
        if N_obs == 1:    
            J = J.reshape(-1,1)
        DJ = GP.DEJ(X_test).reshape(1,dim,dim)
        DJ2 = jnp.einsum('ijk->jki', DJ)
        
        W = epsilon.dot(epsilon.T)

        L = jnp.linalg.cholesky(cov)
        
        
        DL = jacfwd(cov_fun)(X_test)
        
        termJ = jnp.einsum('jki,im->jkm', DJ2, J.T)+jnp.einsum('ji,ikm->jkm', J, DJ)
        
        term1 = jnp.einsum('jik,im->jmk', DL, W)
        term2 = jnp.einsum('jmk,im->jik', term1, L)
        term3 = jnp.einsum('ji,im->jm', L, W)
        term4 = jnp.einsum('jm,imk->jik', term3, DL)
        term11 = term2+term4
        
        term5 = jnp.einsum('ji,im->jm', epsilon, J.T)
        term6 = jnp.einsum('jik,im->jmk', DL, term5)
        term7 = jnp.einsum('mi,ijk->mjk', epsilon, DJ)
        term8 = jnp.einsum('im,mjk->ijk', L, term7)
        term22 = term6+term8
        
        term33 = jnp.einsum('ijk->jik', term8)
        
        G = (J.T+L.dot(epsilon.reshape(-1))).reshape(-1,1)
        G = G.dot(G.T)
        
        return G, termJ+term11+term22+term33
    
    def Ebvp_geodesic(x,y, grid=jnp.linspace(0,1,100), tol=1e-05, method='runge-kutta'):
        
        def chris_symbols(x):
            
            G = GP.Emmf(x)
            
            G_inv = jnp.linalg.inv(G)
            DG = GP.DEmmf(x)
            
            chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                         +jnp.einsum('lij, lm->mji', DG, G_inv) \
                         -jnp.einsum('ijl, lm->mji', DG, G_inv))
            
            return chris
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = chris_symbols(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def Eivp_geodesic(x,v, grid=jnp.linspace(0,1,100), tol=1e-05, method='runge-kutta'):
        
        def chris_symbols(x):
            
            G = GP.Emmf(x)
            
            G_inv = jnp.linalg.inv(G)
            DG = GP.DEmmf(x)
            
            chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                         +jnp.einsum('lij, lm->mji', DG, G_inv) \
                         -jnp.einsum('ijl, lm->mji', DG, G_inv))
            
            return chris
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = chris_symbols(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)        
        
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def ivp_geodesic(x,v, eps = None, grid=jnp.linspace(0,1,1000), tol=1e-05, method='runge-kutta'):
        
        def eq_geodesic(t, y, eps):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            G, DG = GP.Dmmf(gamma, eps)
            G = G+jnp.eye(dim)*1e-2
            
            DG = jnp.einsum('ijk->kij', DG).reshape(dim,-1, order='F')
                        
            return jnp.concatenate((Dgamma, -1/2*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma, Dgamma))))
                    
        def step_fun(yn, time):
        
            #tn, hn = time
            #hn2 = hn/2
            
            #k1 = eq_geodesic(tn, yn, eps)
            #k2 = eq_geodesic(tn+hn2, yn+hn2*k1, eps)
            #k3 = eq_geodesic(tn+hn2, yn+hn2*k2, eps)
            #k4 = eq_geodesic(tn+hn, yn+hn*k3, eps)
            
            #y = yn+hn*(k1+2*k2+2*k3+k4)
            
            #return y, y
        
            tn, hn = time
            y = yn+eq_geodesic(tn,yn, eps)*hn
            
            return y, y
        
        N = len(x)
        
        dt_grid = jnp.hstack((jnp.diff(grid)))
        x0 = jnp.concatenate((x, v), axis=0)
        if eps is None:
            eps = sp.sim_multinormal(mu=jnp.zeros(dim), cov=jnp.eye(dim), dim=1).reshape(dim,1)
        else:
            eps = eps.reshape(dim,1)
        
        _, yt = lax.scan(step_fun, x0, xs=(grid[1:], dt_grid))
        
        yt = jnp.concatenate((x0[jnp.newaxis,...],yt), axis=0)
        
        gamma = yt[:,0:len(x)]
        Dgamma = yt[:,len(x):]
        
        return gamma, Dgamma

    def mmf(X_test):
        
        J = GP.J(X_test).reshape(1,-1)
        
        return J.T.dot(J)
    
    sigman2 = sigman**2
    
    if X_training.ndim == 1:
        X_training = X_training.reshape(1,-1)
        
    dim, N_training = X_training.shape
    
    if y_training.ndim == 1:
        N_obs = y_training.shape[-1]
    else:
        N_obs = 1
    
    if Dm_fun is None and m_fun is not None:
        Dm_fun = vmap(lambda x: grad(m_fun))
    elif m_fun is None and Dm_fun is None:
        Dm_fun = lambda x: jnp.zeros(x.shape[-1])
    
    if m_fun is None:
        m_fun = lambda x: jnp.zeros(x.shape[-1])
        
    if k is None:
        k = km.gaussian_kernel
        
    K11_theta = lambda theta: km.km(X_training, X_training, lambda x,y: k(x,y,*theta))+sigman2*jnp.eye(N_training)
    if grad_K is None:
        grad_K = jacfwd(K11_theta)
    
    if optimize:
        theta = optimize_hyper(theta_init)
        k_fun = lambda x,y: k(x,y,*theta)
    else:
        k_fun = k
    
    if Dk_fun is None:
        Dk_fun = grad(k_fun, argnums=0)
    if DDk_fun is None:
        #DDk_fun = jacfwd(grad(k_fun, argnums=0), argnums=0)
        DDk_fun = jacfwd(jacrev(k_fun, argnums=0), argnums=0)
        
    DDDk_fun = jacfwd(DDk_fun)
    
    K11 = km.km(X_training.T, X_training.T, k_fun)+sigman2*jnp.eye(N_training)+jnp.eye(N_training)*delta_stable
    
    DK11 = jnp.einsum('ijk->jki', vmap(lambda y: vmap(lambda x: Dk_fun(x, y))(X_training.T))(X_training.T))
    
    m_training = m_fun(X_training)
    Dm_training = Dm_fun(X_training)
    
    GP = gp_features() 
    GP.sim_prior = sim_prior
    GP.post_mom = jit(post_mom)
    GP.sim_post = sim_post
    GP.log_ml = jit(log_ml)
    GP.opt = optimize_hyper
    GP.jac_mom = jacobian_mom
    GP.J = sim_jacobian
    GP.Emmf = jit(Emmf)
    GP.mmf = mmf
    GP.DEJ = jit(DEJ)
    GP.Dcov = jit(Dcov)
    GP.DEmmf = jit(DEmmf)
    GP.Dmmf = Dmmf
    GP.geo_Ebvp = Ebvp_geodesic
    GP.geo_Eivp = Eivp_geodesic
    GP.geo_ivp = ivp_geodesic
    
    return GP