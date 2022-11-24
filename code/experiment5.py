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




def model_vs_scale_and_distro(m,n,scatter_distros,scales, scatter=0.1, bg_dist=None, bg_scale=1,prop=0.5,nsamp=10,seed=42,npoints=100):
    """
    detect affine line
    :return:
    """
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    rng = random.default_rng(seed)
    affine_set = sim_affine_set(n,m,model_dist)
    ndistros = len(scatter_distros)
    nscales  = len(scales)
    nfas = np.zeros((ndistros,nscales))
    for i,scatdist in enumerate(scatter_distros):
        nmodel = int(np.ceil(prop*npoints))
        nback  = npoints - nmodel
        seeds = rng.integers(low=1,high=65535,size=nsamp)
        nseeds = len(seeds)
        for seed in seeds:
            model_points = sim_affine_cloud(affine_set, nmodel, model_dist, scatter_distro=scatdist, scatter=scatter)
            back_points = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                if seed == seeds[0]:
                    print(f"\tscatter distro no. {i} scale {s:6.3f} samples {nsamp:3}")
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                lognfa = np.log(max(nfa,1e-40))
                nfas[i,j] += lognfa
    return nfas*(1/nseeds)


def run_experiments():
    nsamp  = 50
    from scipy import stats
    scales = np.arange(0.01,0.41,step=0.01)#np.logspace(-10,-2,base=2,num=40)
    def_scatter = 0.1
    for n in (2,3):
        for m in range(n):
            factors = np.arange(0.0,(n-m)*0.8,step=0.01)
            print(f"\n=======================\nn={n} m={m}")
            print("=======================")
            distros = [build_scatter_distribution(n - m, f) for f in factors]
            x = np.linspace(0,1,100)
#            for i,d in enumerate(distros):
#                plt.figure(i)
#                y = d(10000)
#                plt.hist(y)
#                plt.xlim(0,1.1)
#                plt.show()
            nfas   = model_vs_scale_and_distro(m,n,distros,scales,scatter=def_scatter,nsamp=nsamp,npoints=100)
            ax     = plot_scores_2d(factors,'exponential factor',scales,'analysis scale',nfas,f'NFA vs decay factor n={n} m={m}')

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
