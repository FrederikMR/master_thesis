# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 01:08:27 2021

@author: Frederik
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
    
def x3_fun(x1, x2):
    
    return x1, x2, x1**2-x2**2

class plot_3d_fun(object):
    def __init__(self,
                 N = 100,
                 fig_size = (8,6)):
        
        self.N = N
        self.fig_size = fig_size
        
    def convert_list_to_np(self, Z):
        
        N = len(Z)
        n = len(Z[0])
        Z_new = torch.empty(N, n)
        
        for i in range(N):
            Z_new[i] = Z[i]
            
        return Z_new.detach().numpy()
    
    def cat_tensors(self, Z, dim = 0):
        
        N = len(Z)
        dim = list(Z[0].shape)
        dim[0] = N
        
        Z_tensor = torch.empty(dim)
        
        for i in range(N):
            Z_tensor[i] = Z[i]
        
        return Z_tensor
    
    def plot_means_with_true_shape3d(self, fun, points, *args):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1, x2, x3 = fun(N = self.N)
        ax.plot(x1, x2, x3)
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        ax.scatter3D(x, y, z, color='black')
        
        for arg in args:
            lab = arg[1]
            x = arg[0][0]
            y = arg[0][1]
            z = arg[0][2]
            ax.scatter3D(x, y, z, label=lab, s = 100)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
        plt.legend()
                
        plt.tight_layout()
        
        plt.show()
        
        return
    
    def plot_means_with_true_surface3d(self, fun, points, x1_grid, x2_grid,
                                       xscale=[0,2], yscale=[0,2], zscale=[0,2],
                                       *args):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = self.N)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = self.N)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        ax.scatter3D(x, y, z, color='black')
        
        for arg in args:
            lab = arg[1]
            x = arg[0][0]
            y = arg[0][1]
            z = arg[0][2]
            ax.scatter3D(x, y, z, label=lab, s=100)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
        plt.legend()
        
        ax.set_xlim(xscale[0], xscale[1])
        ax.set_ylim(yscale[0], yscale[1])
        ax.set_zlim(zscale[0], zscale[1])
                
        plt.tight_layout()
        
        plt.show()
        
        return
    
    def plot_geodesic_in_Z_2d(self, *args):
        
        plt.figure(figsize=self.fig_size)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            plt.plot(x, y, '-*', label=lab)
            
        plt.xlabel(r'$x^{1}$')
        plt.ylabel(r'$x^{2}$')
        plt.grid()
        plt.legend()
        plt.title('Geodesic in Z')
        
        plt.tight_layout()

        
        plt.show()
        
        return
    
    def plot_dat_in_Z_2d(self, *args):
        
        plt.figure(figsize=self.fig_size)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            plt.plot(x, y, 'o', label=lab)
            
        plt.xlabel(r'$x^{1}$')
        plt.ylabel(r'$x^{2}$')
        plt.grid()
        plt.legend()
        plt.title('Z-space')
        
        plt.tight_layout()

        
        plt.show()
        
    def plot_mean_in_Z2d(self, points, *args):
        
        plt.figure(figsize=self.fig_size)
        
        plt.scatter(points[:,0], points[:,1])
        
        for arg in args:
            lab = arg[1]
            x = arg[0][0]
            y = arg[0][1]
            plt.scatter(x, y, label=lab, s = 100)
            
        plt.xlabel(r'$x^{1}$')
        plt.ylabel(r'$x^{2}$')
        plt.grid()
        plt.legend()
        plt.title('Z-space')
        
        plt.tight_layout()

        plt.show()
        
        return
    
    def plot_geodesic_in_X_3d(self, fun, x1_grid, x2_grid, *args):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        low_val = 1
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = self.N)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = self.N)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            z = arg[0][:,2]
            ax.plot(x, y, z, label=lab)
            if np.max(z)>1e-3:
                low_val = 0

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        if low_val:
            ax.set_zlim(-2, 2)
        
        ax.legend()
                
        plt.tight_layout()

        plt.show()
        
        return
    
    def plot_geodesic2_in_X_3d(self, *args):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        r = 1
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
        x = r*sin(phi)*cos(theta)
        y = r*sin(phi)*sin(theta)
        z = r*cos(phi)
        
        ax.plot_surface(
            x, y, z,  rstride=1, cstride=1, color='c', alpha=0.1, linewidth=0)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            z = arg[0][:,2]
            ax.plot(x, y, z, label=lab)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
        
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        
        ax.legend()
                
        plt.tight_layout()

        plt.show()
        
        return
    
    def plot_parallel_in_X_3d(self, fun, x1_grid, x2_grid, *args):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        low_val = 1
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = self.N)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = self.N)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            z = arg[0][:,2]
            point = arg[2]
            point_lab = arg[3]
            ax.plot(x, y, z, label=lab)
            ax.scatter3D(point[0], point[1], point[2], label=point_lab, s=50)
            if np.max(z)>1e-3:
                low_val = 0

        ax.set_xlabel(r'$x^{1}$')
        ax.set_xlabel(r'$x^{2}$')
        ax.set_zlabel('z')
        
        if low_val:
            ax.set_zlim(-2, 2)
        
        ax.legend()
                
        plt.tight_layout()

        plt.show()
        
        return
    
    def plot_geodesic3d(self, fun, points, xscale = [-1,1], yscale=[-1,1], 
                                zscale=[-1,1], *args):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1, x2, x3 = fun(N = self.N)
        ax.plot(x1, x2, x3)
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        ax.scatter3D(x, y, z, color='black')
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            z = arg[0][:,2]
            ax.plot(x, y, z, label=lab)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_xlabel(r'$x^{2}$')
        ax.set_zlabel('z')
        plt.legend()
        
        ax.set_xlim(xscale[0], xscale[1])
        ax.set_ylim(yscale[0], yscale[1])
        ax.set_zlim(zscale[0], zscale[1])
                
        plt.tight_layout()
        
        plt.show()
        
    def true_Surface3d(self, fun, x1_grid, x2_grid):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = self.N)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = self.N)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=1.0, linewidth=0)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
                
        plt.tight_layout()
        
        plt.show()
        
    def true_path3d_with_points(self, fun, points, xscale = [-1,1], yscale=[-1,1], 
                                zscale=[-1,1]):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1, x2, x3 = fun(N = self.N)
        ax.plot(x1, x2, x3)
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        ax.scatter3D(x, y, z, color='black', s = 10)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
        
        ax.set_xlim(xscale[0], xscale[1])
        ax.set_ylim(yscale[0], yscale[1])
        ax.set_zlim(zscale[0], zscale[1])
                
        plt.tight_layout()
        
        plt.show()
        
    def plot_1d_hist(self, z, lab):
        
        plt.figure(figsize=self.fig_size)
        n, bins, patches = plt.hist(x=z, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(lab)
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        
    def plot_data_surface_3d(self, x1, x2, x3, title="Surface of Data"):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        ax.plot_trisurf(x1, x2, x3,
                cmap='viridis', edgecolor='none')
        ax.set_title(title)
        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
        
        if np.max(x3)<1e-3:
            ax.set_zlim(-2, 2)
        
        plt.tight_layout()
        
        plt.show()
        
    def plot_data_scatter_3d(self, fun, x1, x2, x3, title="Scatter of Data"):
        
        plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(min(x1), max(x1), num = self.N)
        x2_grid = np.linspace(min(x2), max(x2), num = self.N)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

        ax.set_xlabel(r'$x^{1}$')
        ax.set_ylabel(r'$x^{2}$')
        ax.set_zlabel('z')
        ax.set_title(title)
        
        if np.max(x3)<1e-3:
            ax.set_zlim(-2, 2)
                
        ax.scatter3D(x1, x2, x3, color='black')
                
        plt.tight_layout()
        
        plt.show()
        
    def plot_loss(self, loss, title='Loss function'):
        
        plt.figure(figsize=self.fig_size)
        
        plt.plot(loss)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.title(title)
        
        plt.tight_layout()
        
        plt.show()
        
        
        

        