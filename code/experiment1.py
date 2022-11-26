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

def model_vs_scale_and_npoints(m,n,
                         npointses,
                         scales,
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
        scatter_dist = build_scatter_distribution(n - m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    affine_set_1 = sim_affine_set(n,m,model_dist)
    nscales = len(scales)
    nnp   = len(npointses)
    seeds = rng.integers(low=1, high=65535, size=nsamp)
    nseeds = len(seeds)
    nfas = np.zeros((nnp,nscales))
    for i,npoints in enumerate(npointses):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        t0 = time.time()
        for seed in seeds:
            model_points = sim_affine_cloud(affine_set_1, nmodel, model_dist, scatter_dist, scatter=scatter)
            back_points  = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set_1, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1 # np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (nnp-i)*dt
        print(f'dt={dt:8.2f}s, {rt:8.2f}s to go')
    return  nfas/nseeds

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
        #print('dt=',dtresment()