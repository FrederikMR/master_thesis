#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:01:35 2022

@author: frederik
"""

#%% Sources


#%% MOdules

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
import torch.optim as optim

import numpy as np

import jax.numpy as jnp
from jax import vmap

from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib as mpl

import gaussian_proces as gp
import sp
from VAE_celeba import VAE_CELEBA

#%% Functions

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

#%% Downloading data

img_size = 32
transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_data = datasets.ImageFolder('celeba/',
                                  transform = transform)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False)

images, _ = next(iter(dataloader))

#%% Hyper-parameters

N_sim = 10
sigman = 0.1

eps = sp.sim_multinormal(mu=jnp.zeros(2), cov=jnp.eye(2), dim=N_sim*3*img_size*img_size).reshape(N_sim, 2, 3*img_size*img_size)

#%% Plotting samples

plt.figure(figsize=(8,6))
plt.axis("off")
plt.title("Sample Images")
plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))

#%% Plotting Original image

img_base = jnp.transpose(jnp.asarray(images[0]), axes=(1,2,0))

plt.figure(figsize=(8,6))
plt.imshow((img_base * 255).astype(np.uint8))
plt.title('Original image')
plt.show()

#%% Rotating image

img_base = jnp.transpose(jnp.asarray(images[0]), axes=(1,2,0))

num_rotate = 100
theta = jnp.linspace(0,2*jnp.pi,num_rotate)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)

theta_degrees = theta*180/jnp.pi

rot = []
for v in theta_degrees:
    rot.append(ndimage.rotate(img_base, v, reshape=False))
rot = jnp.stack(rot)

#%% Plotting rotations

rot_plot = np.asarray(jnp.einsum('ijkm->imjk',rot))
rot_plot = torch.from_numpy(rot_plot)
#plt_idx = 2*np.arange(0,99)
#rot_plot2 = rot_plot[plt_idx]
#rot_plot = torch.cat((rot_plot2, rot_plot[-1].reshape(-1,3,img_size,img_size)))

plt.figure(figsize=(8,6))
plt.axis("off")
plt.title("Rotated Images")
plt.imshow(np.transpose(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))
        
#%% Construct data for Gaussian process

X_training = jnp.vstack((x1,x2))
y_training = rot.reshape(num_rotate, -1).T
RMEG = gp.RM_EG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)
RMSG = gp.RM_SG(X_training, y_training, sigman=sigman, k_fun=k_fun, Dk_fun = Dk_fun, DDk_fun = DDk_fun, delta_stable=1e-10)

#%% Loading VAE

#Hyper-parameters
lr = 0.0001
device = 'cpu'

#Loading model
model = VAE_CELEBA().to(device) #Model used
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load('trained_models/celeba/celeba_epoch_50000.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
elbo = checkpoint['ELBO']
rec_loss = checkpoint['rec_loss']
kld_loss = checkpoint['KLD']

model.eval()

#%% Plotting reconstruction for GP

rec_data, _ = RMEG.post_mom(X_training)

rec_data = rec_data.T.reshape(-1,img_size,img_size,3)

rot_plot = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
rot_plot = torch.from_numpy(rot_plot)
#plt_idx = 2*np.arange(0,99)
#rot_plot2 = rot_plot[plt_idx]
#rot_plot = torch.cat((rot_plot2, rot_plot[-1].reshape(-1,3,img_size,img_size)))

plt.figure(figsize=(8,6))
plt.axis("off")
plt.title("Reconstructed Rotated Images for GP")
plt.imshow(np.transpose(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))

#%% Plotting reconstruction for VAE

# Plot some training images
real_batch = torch.Tensor(np.einsum('ijkm->imjk', np.array(rot)))
recon_batch = model(real_batch) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
x_hat = recon_batch[1].detach()

# Plot some training images
plt.figure(figsize=(8,6))
plt.axis("off")
plt.title("Reconstruction Rotated Images for VAE")
plt.imshow(np.transpose(vutils.make_grid(x_hat.to(device), padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))
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

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
ax.legend([legend1, legend2], ['Realised Curvature', 'Curvature for EG'], numpoints = 1)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X[:,:,0], X[:,:,1], secEG, color='orange', alpha=0.2)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of stochastic sectional curvature')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='orange', marker = 'o')
ax.legend([legend2], ['Curvature for EG'], numpoints = 1)

plt.tight_layout()
plt.show()

#%% Geodesics IVP

theta0 = jnp.pi/3

x0 = jnp.array([jnp.cos(theta0), jnp.sin(theta0)])
v0 = jnp.array([0.5, -0.50])

gammaEG, _ = RMEG.geo_ivp(x0,v0)
gammaEG_manifold, _ = RMEG.post_mom(gammaEG.T)
gammaEG_manifold = gammaEG_manifold.T.reshape(-1,img_size,img_size, 3)

gammaSG = vmap(lambda x: RMSG.geo_ivp(x0, v0, x)[0])(eps)

gammaSG_plot = []
for i in range(N_sim):
    if jnp.max(gammaSG[i])<2:
        gammaSG_plot.append(gammaSG[i])
if gammaSG_plot:
    gammaSG_plot = jnp.stack(gammaSG_plot)

gammaSG_manifold, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gammaSG_plot))

#tick_list = [28/2]
fig, ax = plt.subplots(11,1, figsize=(8,6))
ax[0].set_title("IVP Geodesics")
#tick_list_x = [28/2]

rec_data = gammaEG_manifold
rot_plot = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
rot_plot = torch.from_numpy(rot_plot)
plt_idx = 10*np.arange(0,9)
rot_plot2 = rot_plot[plt_idx]
rot_plot = torch.cat((rot_plot2, rot_plot[-1].reshape(-1,3,img_size,img_size)))
ax[0].imshow(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).permute(1, 2, 0))
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_yticks([img_size/2])
ax[0].set_yticklabels([r'Geodesic for $\mathbb{E}\left[\mathbf{G}\right]$'])

for i in range(N_sim):
    rec_data = gammaSG_manifold[i].T.reshape(-1,img_size,img_size, 3)
    rot_plot3 = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
    rot_plot3 = torch.from_numpy(rot_plot3)
    plt_idx = 10*np.arange(0,9)
    rot_plot2 = rot_plot3[plt_idx]
    rot_plot2 = torch.cat((rot_plot2, rot_plot3[-1].reshape(-1,3,img_size,img_size)))
    
    ax[i+1].axes.get_xaxis().set_visible(False)
    ax[i+1].imshow(vutils.make_grid(rot_plot2, padding=2, normalize=True, nrow=10).permute(1, 2, 0))
    ax[i+1].set_yticks([img_size/2])
    ax[i+1].set_yticklabels([r'Sample Geodesic for $\mathbf{G}$'])

plt.figure(figsize=(8,6))

num_rotate = 100
theta = jnp.linspace(0,2*jnp.pi,num_rotate)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)
plt.plot(X_training[0], X_training[1], '-*', label='True Circle', color='blue')

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

num_rotate = 100
theta = jnp.linspace(0,2*jnp.pi,num_rotate)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)
plt.plot(x1, x2, '-*', label='True Circle', color='blue')
plt.axis("equal")

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


img_base = jnp.transpose(jnp.asarray(images[0]), axes=(1,2,0))

num_rotate = 100
theta = jnp.linspace(thetaT,theta0,num_rotate)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)

theta_degrees = theta*180/jnp.pi

gamma_true = []
for v in theta_degrees:
    gamma_true.append(ndimage.rotate(img_base, v, reshape=False))
gamma_true = jnp.stack(gamma_true)[::-1]

load_path = 'rm_computations/celeba_geodesic.pt'
fig, ax = plt.subplots(3,1, figsize=(8,6))
ax[0].set_title("Geodesic cuves and Linear interpolation between images")
img_height = img_size+2
checkpoint = torch.load(load_path)
i = 0

G_plot = checkpoint['G_plot']
arc_length = checkpoint['arc_length']
T = checkpoint['T']

gammaEG, _ = RMEG.geo_bvp(x0,xT)
gammaEG_manifold, _ = RMEG.post_mom(gammaEG.T)
gammaEG_manifold = gammaEG_manifold.T.reshape(-1,img_size,img_size,3)

gammaSG_plot = []
for i in range(N_sim):
    print(i)
    gammaSG, _ = RMSG.geo_bvp(x0, xT, eps[i])
    if jnp.max(gammaSG)<2:
        gammaSG_plot.append(gammaSG)
if gammaSG_plot:
    gammaSG_plot = jnp.stack(gammaSG_plot)

gammaSG_manifold, _ = vmap(RMSG.post_mom)(jnp.einsum('ijk->ikj', gammaSG_plot))

#tick_list = [28/2]
fig, ax = plt.subplots(13,1, figsize=(8,6))
ax[0].set_title("BVP Geodesics")
#tick_list_x = [28/2]

rot_plot = np.asarray(jnp.einsum('ijkm->imjk',gamma_true))
rot_plot10 = torch.from_numpy(rot_plot)
plt_idx = 10*np.arange(0,9)
rot_plot2 = rot_plot10[plt_idx]
rot_plot = torch.cat((rot_plot2, rot_plot10[-1].reshape(-1,3,img_size,img_size)))
ax[0].imshow(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).permute(1, 2, 0))
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_yticks([img_size/2])
ax[0].set_yticklabels([r'True Geodesic'])
#ax[0].set_xticks(['Test'])
#ax[0].set_xticklabels([28/2]) 

rec_data = gammaEG_manifold
rot_plot = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
rot_plot = torch.from_numpy(rot_plot)
plt_idx = 10*np.arange(0,9)
rot_plot2 = rot_plot[plt_idx]
rot_plot = torch.cat((rot_plot2, rot_plot[-1].reshape(-1,3,img_size,img_size)))
ax[1].imshow(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).permute(1, 2, 0))
ax[1].axes.get_xaxis().set_visible(False)
ax[1].set_yticks([img_size/2])
ax[1].set_yticklabels([r'Geodesic for $\mathbb{E}\left[\mathbf{G}\right]$'])

plt_idx = 10*np.arange(0,9)
rot_plot2 = G_plot[plt_idx]
rot_plot = torch.cat((rot_plot2, G_plot[-1].reshape(-1,3,img_size,img_size)))
ax[2].imshow(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).permute(1, 2, 0))
ax[2].axes.get_xaxis().set_visible(False)
ax[2].set_yticks([img_size/2])
ax[2].set_yticklabels([r'Geodesic for VAE'])

for i in range(N_sim):
    rec_data = gammaSG_manifold[i].T.reshape(-1,img_size,img_size, 3)
    rot_plot3 = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
    rot_plot3 = torch.from_numpy(rot_plot3)
    plt_idx = 10*np.arange(0,9)
    rot_plot2 = rot_plot3[plt_idx]
    rot_plot2 = torch.cat((rot_plot2, rot_plot3[-1].reshape(-1,3,img_size,img_size)))
    
    ax[i+3].axes.get_xaxis().set_visible(False)
    ax[i+3].imshow(vutils.make_grid(rot_plot2, padding=2, normalize=True, nrow=10).permute(1, 2, 0))
    ax[i+3].set_yticks([img_size/2])
    ax[i+3].set_yticklabels([r'Sample Geodesic for $\mathbf{G}$'])

plt.figure(figsize=(8,6))

num_rotate = 100
theta = jnp.linspace(0,2*jnp.pi,num_rotate)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)
plt.plot(X_training[0], X_training[1], '-*', label='True Circle', color='blue')

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

num_rotate = 100
theta = jnp.linspace(0,2*jnp.pi,num_rotate)
x1 = jnp.cos(theta)
x2 = jnp.sin(theta)
plt.plot(x1, x2, '-*', label='True Circle', color='blue')
plt.axis("equal")

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

EG_error = jnp.linalg.norm((gamma_true-gammaEG_manifold).reshape(100,img_size*img_size*3), axis=1)
VAE_error = jnp.linalg.norm((gamma_true-jnp.einsum('ijkm->ikmj', jnp.array(G_plot.detach().numpy()))).reshape(100,img_size*img_size*3), axis=1)
SG_error = []
for i in range(N_sim):
    SG_error.append(jnp.linalg.norm((gammaSG_manifold[i].T.reshape(-1,img_size,img_size,3)-gamma_true).reshape(100,img_size*img_size*3), axis=1))

plt.figure(figsize=(8,6))
plt.plot(jnp.linspace(0,1,100), EG_error, '-*', label='GP Geoodesic for EG', color='orange')
plt.plot(jnp.linspace(0,1,100), VAE_error, '-*', label='VAE Geoodesic for EG', color='purple')

for i in range(N_sim-1):
    plt.plot(jnp.linspace(0,1,100), SG_error[i], '-*', color='cyan')

plt.plot(jnp.linspace(0,1,100), SG_error[-1], label='GP Geodesic for SG', color='cyan')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Error')


plt.figure(figsize=(8,6))
plt.plot(jnp.linspace(0,1,100), EG_error, '-*', label='GP Geoodesic for EG', color='orange')

for i in range(N_sim-1):
    plt.plot(jnp.linspace(0,1,100), SG_error[i], '-*', color='cyan')

plt.plot(jnp.linspace(0,1,100), SG_error[-1], label='GP Geodesic for SG', color='cyan')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Error')


#%% SOmething else


#for j in range(T+1):
#    tick_list_x.append(img_height/2+j*img_height)
#    euc_length_x.append('{0:.3f}'.format(torch.norm((G_plot[j]-G_plot[j+T+1]).view(-1)).item()))

#G_plot = checkpoint['G_plot']
#ax[i].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
#ax[i].axes.get_xaxis().set_visible(False)
#ax[i].set_yticks(tick_list)
#ax[i].set_yticklabels(arc_length)
#ax[i].set_xticks(tick_list_x)
#ax[i].set_xticklabels(euc_length_x) 


#rec_data = gammaEG_manifold
#rot_plot = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
#rot_plot = torch.from_numpy(rot_plot)
#plt_idx = 10*np.arange(0,9)
#rot_plot2 = rot_plot[plt_idx]
#rot_plot = torch.cat((rot_plot2, rot_plot[-1].reshape(-1,3,img_size,img_size)))

#plt.figure(figsize=(8,6))
#plt.axis("off")
#plt.title("Reconstructed Rotated Images")
#plt.imshow(np.transpose(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))

#rec_data = gammaEG_manifold
#rot_plot = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
#rot_plot = torch.from_numpy(rot_plot)
#plt_idx = 2*np.arange(0,9)
#rot_plot2 = rot_plot[plt_idx]
#rot_plot = torch.cat((rot_plot2, rot_plot[-1].reshape(-1,3,img_size,img_size)))

#for i in range(N_sim):
#    rec_data = gammaSG_manifold[i].T.reshape(-1,img_size,img_size, 3)
#    rot_plot3 = np.asarray(jnp.einsum('ijkm->imjk',rec_data))
#    rot_plot3 = torch.from_numpy(rot_plot3)
#    plt_idx = 10*np.arange(0,9)
#    rot_plot2 = rot_plot3[plt_idx]
#    rot_plot3 = torch.cat((rot_plot2, rot_plot3[-1].reshape(-1,3,img_size,img_size)))
    
#    rot_plot = torch.cat((rot_plot, rot_plot3))

#plt.figure(figsize=(8,6))
#plt.axis("off")
#plt.title("Reconstructed Rotated Images")
#plt.imshow(np.transpose(vutils.make_grid(rot_plot, padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))















