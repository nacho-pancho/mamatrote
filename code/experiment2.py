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

import matplotlib.cm as cm
from troteplot import *


def model_vs_scale_and_scatter(m,n,scatters,scales,scatter_dist=None, bg_dist=None, bg_scale=1,npoints=100,prop=0.5,nsamp=10,seed=42):
    """
    detect affine line
    :return:
    """
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    rng = random.default_rng(seed)
    affine_set = sim_affine_set(n,m,model_dist)
    nfas = np.zeros((len(scatters),len(scales)))
    for i,scat in enumerate(scatters):
        nmodel = int(np.ceil(prop*npoints))
        nback  = npoints - nmodel
        seeds = rng.integers(low=1,high=65535,size=nsamp)
        nseeds = len(seeds)
        for seed in seeds:
            model_points = sim_affine_cloud(affine_set, nmodel, model_dist, scatter_dist, scatter=scat)
            back_points = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                #if seed == seeds[0]:
                    #print(f"\tscatter {scat:6.3f} scale {s:6.3f} samples {nsamp:3}")
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1# lognfa
    return nfas*(1/nseeds)

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
    nsamp  = 50
    Ns     = np.round(np.logspace(6,10,base=2,num=25)).astype(int)
    scales = np.logspace(-10,-2,base=2,num=40)
    scatters = np.logspace(-2,-1,num=40,base=10)
    def_scatter = 0.1
    for n in (2, 3):
        for m in range(n):
            print(f"\n=======================\nn={n} m={m}")
            print("=======================")
            fbase  = (f'NFA vs scale and scatter n={n} m={m}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt'):
                nfas = model_vs_scale_and_scatter(m, n, scatters, scales, nsamp=nsamp)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', Ns)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(scatters,'model scatter',scales,'analysis scale',nfas,f'NFA vs scale and scatter n={n} m={m}')

