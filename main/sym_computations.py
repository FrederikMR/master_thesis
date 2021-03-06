#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:06:38 2022

@author: frederik
"""

#%% Modules

#Adds the source library to Python Path
import sys
sys.path.insert(1, '../src/')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from sympy.plotting import plot_implicit

import rm_sympy as dif_geo_sympy

import sympy as sym

#%% Seed

np.random.seed(2712)

#%% Hyper-parameter

np.random.seed(2712)

sigma = 1.0
mu = 0.0
n_points = 100
grid0 = -2
grid1 = 2
N_sim = 10

x = np.linspace(grid0,grid1,n_points)
y = np.linspace(grid0,grid1,n_points)
X, Y = np.meshgrid(x, y)

x1, x2 = sym.symbols('x1 x2')
v1, v2 = sym.symbols('v1 v2')
x = sym.Matrix([x1, x2])
a1, a2 = sym.symbols('a1 a2')
G = None

param_fun = sym.Matrix([x1, x2, a1*x1**2+a2*x2**2])

#%% Initial Computations

if G is None:
    _ = dif_geo_sympy.compute_mmf(param_fun, x, print_latex = True)
_ = dif_geo_sympy.chris_symbols(x, param_fun=param_fun, G=G, print_latex=True)
_ = dif_geo_sympy.eq_geodesics(x, param_fun=param_fun, G=G, print_latex=True)
_ = dif_geo_sympy.sectional_curvature2d(x, param_fun=param_fun, G=G, print_latex=True)


#%% Christoffel symbols

chris = dif_geo_sympy.chris_symbols(x, param_fun=param_fun, G=G, print_latex=False)
EG = sym.Matrix([[1+4*x1**2, 0], [0, 1+4*x2**2]])
Echris = dif_geo_sympy.chris_symbols(x, param_fun=param_fun, G=EG, print_latex=False)
positions = (np.arange(8) + 1).reshape(2,2,2)

x1_subs = 1.0
x2_subs = 1.0

sim = 1000
coef = np.random.normal(mu, sigma, [sim, 2])
val = np.zeros((2,2,2,sim))
for i in range(sim):
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                    val[i1,i2,i3,i] = chris[i1,i2,i3].subs(a1, coef[i,0]).subs(a2, coef[i,1]).subs(x1, x1_subs).subs(x2, x2_subs)

xlabel = []
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            xlabel.append(r'$\Gamma_{{{}}}^{}$'.format(str(i1+1)+str(i2+1),i3+1))

fig = plt.figure(figsize =(8, 6))
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            plt.boxplot(val[i1,i2,i3], positions=[positions[i1,i2,i3]])
            plt.plot([positions[i1,i2,i3]], [Echris[i1,i2,i3].subs(x1, x1_subs).subs(x2, x2_subs)], 'rs')

plt.xticks(positions.reshape(-1), xlabel)
plt.title(r'Stochastic Christoffel symbols at x1={} and x2={}'.format(x1_subs, x2_subs))


#%% Plot realisations as well as expected parametrization (NOT EQUAL TO EXPECTED METRIC)

EZ = np.zeros((n_points, n_points))
coef = np.random.normal(mu, sigma, [N_sim, 2])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, EZ, color='red')

for i in range(np.min((N_sim,10))):
    Z = coef[i][0]*(X)+coef[i][1]*(Y)
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.4)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$a^{1}\left(x^{1}\right)^{2}+a^{2}\left(x^{2}\right)^{2}$')
ax.set_title('Stochastic Paraboloid')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend1, legend2], ['Expected parametrization', 'Realisations'], numpoints = 1)


plt.tight_layout()
plt.show()

#%% Expected Metric

EG = sym.Matrix([[2,0], [0,2]])
_ = dif_geo_sympy.chris_symbols(x, G=EG, print_latex=True)
_ = dif_geo_sympy.eq_geodesics(x, G=EG, print_latex=True)
_ = dif_geo_sympy.sectional_curvature2d(x, G=EG, print_latex=True)

#%% Stochastic Metric

SG, _ = dif_geo_sympy.compute_mmf(param_fun, x)

#%% Indicatrix

plt.rcParams['figure.figsize'] = 8,6
center = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

G = EG.subs(x1,center[0][0]).subs(x2,center[0][1])
ellipse_fun = lambda v1,v2: G[0,0]*(v1**2)+G[1,1]*(v2**2)+2*v1*v2*G[0,1]-1
plot1 = plot_implicit(ellipse_fun(v1,v2), (v1, -2, 2), (v2, -2, 2),
                      axis_center=(center[0][0], center[0][1]),
                      line_color='red', show=False)
for i in range(len(center)):
    G = EG.subs(x1,center[i][0]).subs(x2,center[i][1])
    ellipse_fun = lambda v1,v2: G[0,0]*(v1**2)+G[1,1]*(v2**2)+2*v1*v2*G[0,1]-1
    plot1 = plot_implicit(ellipse_fun(v1,v2), (v1, -2, 2), (v2, -2, 2),
                          axis_center=(center[i][0], center[i][1]),
                          line_color='red', show=False,
                          xlabel=r'$v^{1}$', ylabel=r'$v^{2}$')
    for j in range(N_sim):
        G = SG.subs(a1,coef[j][0]).subs(a2,coef[j][1])
        G = G.subs(x1,center[i][0]).subs(x2,center[i][1])
        ellipse_fun = lambda v1,v2: G[0,0]*(v1**2)+G[1,1]*(v2**2)+2*v1*v2*G[0,1]-1
        plot2 = plot_implicit(ellipse_fun(v1,v2), (v1, -2, 2), (v2, -2, 2),
                              axis_center=(center[i][0], center[i][1]),
                              line_color='cyan', alpha=0.2, show=False)
        plot1.append(plot2[0])
    
    plot1.show()

#%% Geodesics

y0 = np.array([1.0,1.0])
yT = np.array([-1.0,1.0])

N = 1000
n_grid = 100
y_init_grid = np.zeros((4, n_grid))
coef2 = np.random.normal(mu, sigma, [N, 2])
Egamma = 0.0
EgammaZ = 0.0
count = 0.0
for i in range(N):
    print(i)
    G = SG.subs(a1, coef2[i][0]).subs(a2, coef2[i][1])
    ySG, v = dif_geo_sympy.bvp_geodesic(y0, yT, n_grid, y_init_grid, x, G = G)
    if ySG.shape[-1]==100:
        count += 1
        Egamma += ySG
        EgammaZ += coef2[i][0]*ySG[0]**2+coef2[i][1]*ySG[1]**2
Egamma /= count
EgammaZ /= count

n_grid = 100
y_init_grid = np.zeros((4, n_grid))
EG = sym.Matrix([[1+4*x1**2,0], [0,1+4*x2**2]])
yEG, v = dif_geo_sympy.bvp_geodesic(y0, yT, n_grid, y_init_grid, x, G = EG)

fig, ax = plt.subplots(1,2,figsize=(8,6))
ax[0].plot(np.linspace(0,1,yEG.shape[-1]), yEG[0], color='red')
ax[1].plot(np.linspace(0,1,yEG.shape[-1]), yEG[1], color='red')
ySG_ls = []
Z_ls = []
for i in range(N_sim):
    G = SG.subs(a1, coef[i][0]).subs(a2, coef[i][1])
    Z_ls.append(coef[i][0]*(X**2)+coef[i][1]*(Y**2))
    ySG, v = dif_geo_sympy.bvp_geodesic(y0, yT, n_grid, y_init_grid, x, G = G)
    ySG_ls.append(ySG)
    ax[0].plot(np.linspace(0,1,ySG.shape[-1]), ySG[0], color='cyan', alpha=0.4)
    ax[1].plot(np.linspace(0,1,ySG.shape[-1]), ySG[1], color='cyan', alpha=0.4)

ax[0].plot(np.linspace(0,1,ySG.shape[-1]), Egamma[0], color='black', alpha=0.4)
ax[1].plot(np.linspace(0,1,ySG.shape[-1]), Egamma[1], color='black', alpha=0.4)
ax[0].set_xlabel(r'$t$')
ax[0].set_title(r'$\gamma^{1}(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_title(r'$\gamma^{2}(t)$')
ax[0].grid()
ax[1].grid()

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')

for i in range(N_sim):
    ax.plot(ySG_ls[i][0], ySG_ls[i][1], coef[i][0]*ySG_ls[i][0]**2+coef[i][1]*ySG_ls[i][1]**2, color='red', alpha=1.0, linewidth=3.0)
    ax.plot_surface(X, Y, Z_ls[i], color='cyan', alpha=0.1)

ax.plot(Egamma[0], Egamma[1], EgammaZ, c='black')
ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel(r'$\gamma$')
ax.set_title('Realisations of stochastic geodesics')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
ax.legend([legend1, legend2], ['Realisations', 'Sample mean of geodesics'], numpoints = 1)

#%% Curvature

sec = dif_geo_sympy.sectional_curvature2d(x, G=EG)
if sec == 0.0:
    Esec = np.zeros((n_points, n_points))
else:
    Esec = sec(X,Y)

sec = dif_geo_sympy.sectional_curvature2d(x, G=SG)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Esec, color='red')

for i in range(N_sim):
    sec_coef = sec.subs(a1, coef[i][0]).subs(a2, coef[i][1])
    sec_coef = sym.lambdify(x, sec_coef, modules='numpy')
    sec_SG = sec_coef(X,Y)
    ax.plot_surface(X, Y, sec_SG, color='cyan', alpha=0.4)

ax.set_xlabel(r'$x^{1}$')
ax.set_ylabel(r'$x^{2}$')
ax.set_zlabel('Sectional Curvature')
ax.set_title('Realisations of Stochastic Riemannian Manifolds')
legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='red', marker = 'o')
legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='cyan', marker = 'o')
ax.legend([legend1, legend2], ['Curvature for Expected Metric', 'Realised Curvature'], numpoints = 1)


plt.tight_layout()
plt.show()



#%% Curvature 2
N_sim = 10000
coef = np.random.normal(mu, sigma, [N_sim, 2])

K_SG = dif_geo_sympy.sectional_curvature2d(x, param_fun=param_fun, print_latex=False)
K_EG = dif_geo_sympy.sectional_curvature2d(x, G=EG, print_latex=False)

K_SG = K_SG.subs(x1, 0.0).subs(x2,0.0)
K_EG = float(K_EG.subs(x1, 0.0).subs(x2,0.0))

K_SG_list = []
for i in range(N_sim):
    K_SG_list.append(float(K_SG.subs(a1, coef[i,0]).subs(a2,coef[i,1])))

plt.figure(figsize=(8,6))
result = plt.hist(K_SG_list, bins=20, color='c', edgecolor='k', alpha=0.65)
plt.axvline(K_EG, color='k', linestyle='dashed', linewidth=1)
plt.axvline(np.mean(K_SG_list), color='r', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(K_SG_list)*1.1, max_ylim*0.9, 'Sample Mean: {:.2f}'.format(np.mean(K_SG_list)))
plt.text(K_EG*1.1, max_ylim*0.6, 'Expected Curvature: {:.2f}'.format(K_EG))


#MEAN = 0, VAR=16, DISTRIBUTION 17/4*(Q-R), where Q=((4a)+b)**2, R=((4a)-b)**2






