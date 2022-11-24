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

import matplotlib.cm as cm


def model_vs_scale(m,n,N,seed=42):
    """
    detect affine line
    :return:
    """
    rng = random.default_rng(seed)
    distro0 = lambda x: rng.uniform(size=x, low=0, high=1)
    affine_set  = sim_affine_set(n,m,distro0)
    d1 = lambda x: rng.uniform(size=x,low=-2,high=2)
    d2 = lambda x: rng.uniform(size=x,scale=0.2)
    test_points = sim_affine_cloud(affine_set, N, d1,d2)
    mm = np.min(test_points)
    MM = np.max(test_points)
    test_points = np.concatenate((test_points, (MM-mm)*random.rand(10*N, 2)+mm))
    x_0, V, W = affine_set
    a = x_0
    b = x_0 + V[0]
    c = x_0 + W[0]

    scale = 0.5
    affine_set2 = sim_affine_set(n, m, distro0)
    points = np.copy(test_points[:2])
    points += rng.normal(0, 0.01, size=points.shape)
    affine_set3 = build_affine_set(points)
    nfa1 = nfa_ks(test_points, affine_set, m, m+1, distance_to_affine, scale)
    nfa2 = nfa_ks(test_points, affine_set2, m, m+1, distance_to_affine, scale)
    nfa3 = nfa_ks(test_points, affine_set3, m, m+1, distance_to_affine, scale)
    print(f'Scale {scale}')
    print(f'Red NFA {nfa1}')
    print(f'Green NFA {nfa2}')
    print(f'Blue NFA {nfa3}')
    x_0, V, W = affine_set
    a1 = x_0
    b1 = x_0 + V[0]
    c1 = x_0 + W[0]
    x_0, V, W = affine_set2
    a2 = x_0
    b2 = x_0 + V[0]
    c2 = x_0 + W[0]
    x_0, V, W = affine_set3
    a3 = x_0
    b3 = x_0 + V[0]
    c3 = x_0 + W[0]
    nfas = []
    scales = []
    for s in range(20):
        scale = 0.025*s
        nfa = nfa_ks(test_points, affine_set, m, m+1, distance_to_affine, scale)
        nfas.append(-np.log(nfa))
        scales.append(scale)
    plt.semilogy(scales, nfas)
    plt.xlabel('scale')
    plt.ylabel('logNFA')
    plt.show()

def model_vs_scale_and_npoints(m,n,Ns,scales, prop=0.5, scatter_dist=None, bg_dist=None, bg_scale=1, scatter=0.1,seed=42,nsamp=10):
    """
    detect affine line
    :return:
    """
    print("model vs scale and number of points")
    print("scales:",scales)
    print("number of points:",Ns)
    rng = random.default_rng(seed)
    if scatter_dist is None:
        scatter_dist = build_background_distribution(n-m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    affine_set = sim_affine_set(n,m,model_dist)
    nfas = np.zeros((len(Ns),len(scales)))
    for i,N in enumerate(Ns):
        nmodel = int(prop*N)
        nback  = N - nmodel
        seeds = rng.integers(low=1,high=65535,size=nsamp)
        nseeds = len(seeds)
        t0 = time.time()
        for seed in seeds:
            model_points = sim_affine_cloud(affine_set, nmodel, model_dist, scatter_dist, scatter=scatter)
            back_points  = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                if seed == seeds[0]:
                    print(f"\tN {N:6} scale {s:6.3f} samples {nsamp:3}")
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                lognfa = np.log(nfa)
                #print(f'N={N:8} scale={s:8.6f} NFA={lognfa:8.6f}')
                nfas[i,j] += np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (len(Ns)-i)*dt
        #print('dt=',dt,'remaining time=',rt)
    return  nfas/nseeds


def model_vs_scale_and_proportion(m,n,N,props,scales,scatter_dist=None, bg_dist=None, bg_scale=1, scatter=0.1, nsamp=10, seed=42):
    """
    detect affine line
    :return:
    """
    print("model vs scale and proportion of foreground/background points")
    print("scales:",scales)
    print("prportions:",props)

    if scatter_dist is None:
        scatter_dist = build_background_distribution(n-m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    rng = random.default_rng(seed)

    affine_set = sim_affine_set(n,m,model_dist)
    nfas = np.zeros((len(props),len(scales)))
    for i,p in enumerate(props):
        nmodel = int(np.ceil(p*N))
        nback  = N - nmodel
        seeds = rng.integers(low=1,high=65535,size=nsamp)
        nseeds = len(seeds)
        for seed in seeds:
            model_points = sim_affine_cloud(affine_set, nmodel, model_dist, scatter_dist, scatter=scatter)
            back_points = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                if seed == seeds[0]:
                    print(f"\tprop {p:6.3f} scale {s:6.3f} samples {nsamp:3}")
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                lognfa = np.log(max(nfa,1e-40))
                #print(f'N={N:8} scale={s:8.6f} NFA={lognfa:8.6f}')
                nfas[i,j] += lognfa
    return nfas*(1/nseeds)

def model_vs_scale_and_scatter(m,n,N,scatters,scales,scatter_dist=None, bg_dist=None, bg_scale=1,prop=0.5,nsamp=10,seed=42):
    """
    detect affine line
    :return:
    """
    print("model vs scale and different model scatters")
    print("scales:",scales)
    print("scatters:",scatters)

    if scatter_dist is None:
        scatter_dist = build_background_distribution(n-m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    rng = random.default_rng(seed)
    affine_set = sim_affine_set(n,m,model_dist)
    nfas = np.zeros((len(scatters),len(scales)))
    for i,scat in enumerate(scatters):
        nmodel = int(np.ceil(prop*N))
        nback  = N - nmodel
        seeds = rng.integers(low=1,high=65535,size=nsamp)
        nseeds = len(seeds)
        for seed in seeds:
            model_points = sim_affine_cloud(affine_set, nmodel, model_dist, scatter_dist, scatter=scat)
            back_points = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                if seed == seeds[0]:
                    print(f"\tscatter {scat:6.3f} scale {s:6.3f} samples {nsamp:3}")
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                lognfa = np.log(max(nfa,1e-40))
                #print(f'N={N:8} scale={s:8.6f} NFA={lognfa:8.6f}')
                nfas[i,j] += lognfa
    return nfas*(1/nseeds)

def plot_scores_2d(x,xlabel,y,ylabel,nfa,title):
    X, Y = np.meshgrid(x,y)
    Z = nfa.T
    fig = plt.figure(figsize=(10,10))
    ax3 = fig.add_subplot(projection='3d')
    cmap = ListedColormap(["red", "red", "blue", "blue"])
    ax3.plot_surface(X,Y,Z, edgecolor='black', lw=0.25,alpha=0.5,cmap=cmap,vmin=-10,vmax=10)
    ax3.set(xlabel=xlabel,ylabel=ylabel,zlabel='-log(NFA)',title=title)
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

    # cluster in 2D
    m      = 0 # affine space dimension
    n      = 2 # ambient dim
    #model_vs_scale(m,n,N)
    nsamp  = 50
    #Ns     = np.arange(N//10,N+N//10,step=N//10)
    Ns     = np.round(np.logspace(6,10,base=2,num=25)).astype(int)
    scales = np.logspace(-10,-2,base=2,num=25)
    scatters = np.logspace(-2,-1,num=25,base=10)
    def_scatter = 0.1
    nfas   = model_vs_scale_and_scatter(m,n,N,scatters,scales,nsamp=nsamp)
    ax     = plot_scores_2d(scatters,'model scatter',scales,'analysis scale',nfas,'NFA vs scatter and proportion on 2D cluster')
    #
    # line in 2D
    #
    n = 2
    m = 1
    nfas = model_vs_scale_and_scatter(m,n,N,scatters,scales,nsamp=nsamp)
    ax   = plot_scores_2d(scatters,'model scatter',scales,'analysis scale',nfas,'NFA vs scatter and proportion on 2D line')
    #
    # cluster in 3D
    n    = 3
    m    = 0 # affine space dimension
    nfas = model_vs_scale_and_scatter(m,n,N,scatters,scales,nsamp=nsamp)
    ax   = plot_scores_2d(scatters,'model scatter',scales,'analysis scale',nfas,'NFA vs scatter and proportion on 3D cluster')
    #
    # line in 3D
    #
    m = 1
    n = 3
    nfas = model_vs_scale_and_scatter(m,n,N,scatters,scales,nsamp=nsamp)
    ax   = plot_scores_2d(scatters,'model scatter',scales,'analysis scale',nfas,'NFA vs scatter and proportion on 3D line')
    #
    # plane in 3D
    #
    m = 2
    n = 3
    nfas = model_vs_scale_and_scatter(m,n,N,scatters,scales,nsamp=nsamp)
    ax   = plot_scores_2d(scatters,'model scatter',scales,'analysis scale',nfas,'NFA vs scatter and proportion on 3D plane')
