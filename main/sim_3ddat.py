# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:44:58 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
"""

#%% Modules

#Modules
import numpy as np
import jax.numpy as jnp

import pandas as pd 
import torch
import sp

#%% Sim class

N_sim = 100 #Number of simulated points



def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1, x2, x1**2-x2**2

class sim_3d_fun(object):
    def __init__(self,
                 x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = 50000,
                 name_path = 'para_data.csv'):
        
        self.x1_fun = x1_fun
        self.x2_fun = x2_fun
        self.x3_fun = x3_fun
        self.N_sim = N_sim
        self.name_path = name_path
        
    def sim_3d(self):
    
        #np.random.seed(self.seed)
        x1 = self.x1_fun(self.N_sim)
        x2 = self.x2_fun(self.N_sim)
        
        x1, x2, x3 = self.x3_fun(x1, x2)
        
        df = np.vstack((x1, x2, x3))
        
        pd.DataFrame(df).to_csv(self.name_path)
        
        return
    
    def read_data(self):
        
        df = pd.read_csv(self.name_path, index_col=0)

        dat = torch.Tensor(df.values)
        
        dat = torch.transpose(dat, 0, 1)
        
        return dat


#%% Function for simulating plane (R2) in R3

N_training = 40

name_path = 'sim_data/plane.csv' #Path/file_name

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 1.5, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
x3 = jnp.zeros_like(x1_mesh)

df = np.vstack((x1_mesh.reshape(-1), x2_mesh.reshape(-1), x3.reshape(-1)))

pd.DataFrame(df).to_csv(name_path)


#%% Function for simulating parabolic

name_path = 'sim_data/paraboloid.csv' #Path/file_name

N_training = 40

x1 = jnp.linspace(-1.5, 1.5, N_training)
x2 = jnp.linspace(-1.5, 1.5, N_training)
x1_mesh, x2_mesh = jnp.meshgrid(x1,x2)
x3 = x1_mesh**2+x2_mesh**2

df = np.vstack((x1_mesh.reshape(-1), x2_mesh.reshape(-1), x3.reshape(-1)))

pd.DataFrame(df).to_csv(name_path)


#%% Function for simulating circle in R^3

N_training = 1000

name_path = 'sim_data/circle.csv' #Path/file_name
sigman = 0.1
noise = sp.sim_multinormal(mu=jnp.zeros(3), cov=(sigman**2)*jnp.eye(3), dim=N_training).reshape(N_training, 3)

#sigman = 0.0
#noise = jnp.zeros((N_training, 3))

theta = jnp.linspace(0,2*jnp.pi,N_training)

x1 = jnp.cos(theta)+noise[:,0]
x2 = jnp.sin(theta)+noise[:,1]
x3 = noise[:,2]

df = np.vstack((x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)))
        
pd.DataFrame(df).to_csv(name_path)
