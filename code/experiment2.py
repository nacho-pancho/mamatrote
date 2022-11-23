#!/usr/bin/env python3

import time

import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from  trotelib import *
from troteplot import *

import matplotlib.cm as cm


def plot_example(affine_set_1,affine_set_2,points_1,points_2,ran):
    fig = plt.figure(figsize=(12,12))
    n = points_1.shape[1]
    if n == 2:
        ax = fig.add_subplot()
        plot_set(ax, affine_set_1, color1='red', color2='red')
        plot_set(ax, affine_set_2, color1='blue', color2='blue')
        ax.scatter(points_1[:,0],points_1[:,1],color='orange')
        ax.scatter(points_2[:,0],points_2[:,1],color='cyan')
        #ax.xlim(-ran,ran)
        #ax.ylim(-ran,ran)
    elif n == 3:
        ax = fig.add_subplot(projection='3d')
        plot_set(ax, affine_set_1, color1='red', color2='red')
        plot_set(ax, affine_set_2, color1='blue', color2='blue')
        ax.scatter(points_1[:,0],points_1[:,1],points_1[:,2], color='orange')
        ax.scatter(points_2[:,0],points_2[:,1],points_1[:,2], color='cyan')
        ax.view_init(elev=70,azim=120)
        #ax.xlim(-ran,ran)
        #ax.ylim(-ran,ran)
        #ax.zlim(-ran,ran)



def parallel_vs_distance(m,n,
                         distances,
                         scales,
                         npoints=100,
                         prop=0.5,
                         scatter_dist=None,
                         bg_dist=None,
                         bg_scale=1,
                         scatter=0.1,
                         seed=42,
                         nsamp=10):
    """
    detect affine line
    :return:
    """
    rng = random.default_rng(seed)
    if scatter_dist is None:
        scatter_dist = build_background_distribution(n-m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    affine_set_1 = sim_affine_set(n,m,model_dist)
    nscales = len(scales)
    ndist   = len(distances)
    seeds = rng.integers(low=1, high=65535, size=nsamp)
    nseeds = len(seeds)
    nfas = np.zeros((ndist,nscales))
    for i,dist in enumerate(distances):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        t0 = time.time()
        affine_set_2 = build_affine_set_relative_to(affine_set_1, dist=dist, angle=0)
        for seed in seeds:
            model1_points = sim_affine_cloud(affine_set_1, nmodel, model_dist, scatter_dist, scatter=scatter)
            model2_points = sim_affine_cloud(affine_set_2, nmodel, model_dist, scatter_dist, scatter=scatter)
            back_points  = bg_dist((nback, n))
            model_points = np.concatenate((model1_points,model2_points))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set_1, m, m+1, distance_to_affine, s)
                if seed == seeds[0]:
                    #print(f"\tdist {dist:6} scale {s:6.3f} samples {nsamp:3}  log(nfa) {np.log10(nfa):8.4f}")
                    if dist == distances[-1]:
                        plot_example(affine_set_1,affine_set_2,model1_points,model2_points,ran=1)
                        plt.savefig(f"cloud_n_{n}_m{m}_dist_{dist:06.4f}.svg")
                        plt.close()
                lognfa = np.log(nfa)
                #print(f'N={N:8} scale={s:8.6f} NFA={lognfa:8.6f}')
                nfas[i,j] += np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (ndist-i)*dt
        print(f'dt={dt:8.2f}s, {rt:8.2f}s to go')
    return  nfas/nseeds


def parallel_vs_angle(m,n,
                         angles,
                         scales,
                         npoints=100,
                         prop=0.5,
                         scatter_dist=None,
                         bg_dist=None,
                         bg_scale=1,
                         scatter=0.1,
                         seed=42,
                         nsamp=10):
    """
    detect affine line
    :return:
    """
    rng = random.default_rng(seed)
    if scatter_dist is None:
        scatter_dist = build_background_distribution(n-m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    affine_set_1 = sim_affine_set(n,m,model_dist)
    nscales = len(scales)
    nang   = len(angles)
    seeds = rng.integers(low=1, high=65535, size=nsamp)
    nseeds = len(seeds)
    nfas = np.zeros((nang,nscales))
    for i,ang in enumerate(angles):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        t0 = time.time()
        affine_set_2 = build_affine_set_relative_to(affine_set_1, dist=0, angle=ang)
        for seed in seeds:
            model1_points = sim_affine_cloud(affine_set_1, nmodel, model_dist, scatter_dist, scatter=scatter)
            model2_points = sim_affine_cloud(affine_set_2, nmodel, model_dist, scatter_dist, scatter=scatter)
            back_points  = bg_dist((nback, n))
            model_points = np.concatenate((model1_points,model2_points))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set_1, m, m+1, distance_to_affine, s)
                if seed == seeds[0]:
                    #print(f"\tdist {dist:6} scale {s:6.3f} samples {nsamp:3}  log(nfa) {np.log10(nfa):8.4f}")
                    if ang == angles[-1]:
                        plot_example(affine_set_1,affine_set_2,model1_points,model2_points,ran=1)
                        plt.savefig(f"cloud_n_{n}_m{m}_ang_{ang:06.4f}.svg")
                        plt.close()
                lognfa = np.log(nfa)
                #print(f'N={N:8} scale={s:8.6f} NFA={lognfa:8.6f}')
                nfas[i,j] += np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (nang-i)*dt
        print(f'dt={dt:8.2f}s, {rt:8.2f}s to go')
    return  nfas/nseeds


def plot_scores_2d(x,xlabel,y,ylabel,nfa,title):
    from matplotlib.colors import LightSource

    X, Y = np.meshgrid(x,y)
    Z = nfa.T
    fig = plt.figure(figsize=(10,10))
    ax3 = fig.add_subplot(projection='3d')
    thresmap = ListedColormap(["red", "red", "blue", "blue"])

    ls = LightSource(250, 70,hsv_max_val=1) # azimuth, alt
    colors = thresmap(Z)
    Znorm = Z-np.min(Z)
    Znorm  /= np.max(Znorm)
    shaded_colors = ls.shade_rgb(colors,Znorm,vert_exag=0.05)
    ax3.plot_surface(X,Y,Z, edgecolor='black', lw=0.25,facecolors=shaded_colors,antialiased=False,shade=False)
    ax3.set(xlabel=xlabel,ylabel=ylabel,zlabel='-log(NFA)',title=title)
    ax3.view_init(elev=45,azim=120)
    fname = title.replace(' ','_').lower()
    plt.savefig(f'{fname}_3d.svg')
    plt.close(fig)

    #fig = plt.figure(figsize=(10,10))
    #plt.contour(X, Y, Z, cmap=cmap)
    #plt.title(title)
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #plt.grid(True)
    #plt.colorbar()
    #plt.savefig(f'{fname}_contour.svg')
    #plt.close(fig)



def run_experiments():
    # cluster in 2D
    m      = 0 # affine space dimension
    n      = 2 # ambient dim
    #model_vs_scale(m,n,N)
    nsamp  = 50
    #Ns     = np.arange(N//10,N+N//10,step=N//10)
    Ns     = np.round(np.logspace(6,10,base=2,num=25)).astype(int)
    scales = np.arange(0.01,0.41,step=0.01)#np.logspace(-10,-2,base=2,num=40)
    props  = np.arange(0.05,1.05,step=0.05)
    scatters = np.logspace(-2,-1,num=25,base=10)
    def_scatter = 0.1
    #distances = def_scatter*np.power(2.0,np.arange(-1,3,step=0.1))
    distances = def_scatter*np.arange(0.5,8.5,step=0.125)
    if False:
        for n in (2,3):
            for m in range(n):
                print(f"\n=======================\nn={n} m={m}")
                print("=======================")
                nfas   = parallel_vs_distance(m,n,distances,scales,scatter=def_scatter,nsamp=nsamp)
                ax     = plot_scores_2d(distances,'distance',scales,'analysis scale',nfas,f'NFA vs distance n={n} m={m}')
    angles = np.arange(0,np.pi,step=np.pi/30)
    for n in (2,3):
        for m in range(n):
            print(f"\n=======================\nn={n} m={m}")
            print("=======================")
            nfas   = parallel_vs_angle(m,n,angles,scales,scatter=def_scatter,nsamp=nsamp,npoints=10)
            ax     = plot_scores_2d(angles,'angles',scales,'analysis scale',nfas,f'NFA vs angle n={n} m={m}')

#==========================================================================================

import argparse

if __name__ == "__main__":
    plt.close('all')
    #
    # command line arguments
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxn", type=int, default=1000,
                    help="max number of points to simmulate")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="Proportion of scale of dispersion from affine set to scale of global point cloud")
    args = vars(ap.parse_args())
    N    = args["maxn"]
    scatter = args["scatter"]
    m = 1
    n = 2
    N = 100
    #test_relative(m,n,N)
    run_experiments()