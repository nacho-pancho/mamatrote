#!/usr/bin/env python3

import time
import os

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
    rng = random.default_rng(seed)
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m)
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
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1 # np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (len(Ns)-i)*dt
        #print('dt=',dt,'remaining time=',rt)
    return  nfas/nseeds

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
    scatter = args["scatter"]
    nsamp  = 25
    Ns     = np.round(np.logspace(6,10,base=2,num=40)).astype(int)
    scales = np.logspace(-10,-2,base=2,num=40)
    for n in (2,3):
        for m in range(n):
            print(f"\n=======================\nn={n} m={m}")
            print("=======================")
            fbase  = (f'NFA vs scale and npoints n={n} m={m}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt'):
                nfas = model_vs_scale_and_npoints(m,n,Ns,scales,nsamp=nsamp)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', Ns)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax = plot_scores_img(Ns, 'number of points', scales, 'analysis scale', nfas,
                                f'NFA vs scales and npoints n={n} m={m}')
