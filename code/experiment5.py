#!/usr/bin/env python3
"""
This experiment investigates the ability to detect affine structures
as a function of:
a) the analysis scale, which is a parameter of the framework and
b) the scatter distribution, ranging from the uniform (default) distribution to a 1/x^2 one

The other problem parameters are:
* scatter distance 0.1
* proportion of model/background points is 50/50.
* number of points defaults to 100
  distance to the affine set is uniform regardless of the dimension
* the experiment is repeated 10 times for 10 different random seeds

Note: the target structure parameters are known (perfectly).

"""

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




def model_vs_scale_and_distro(m,n,scatter_distros,scales, scatter=0.1, bg_dist=None, bg_scale=1,prop=0.5,nsamp=10,seed=42,npoints=100):
    """
    see wheter we detect the structure or not depending on how concentrated the points are around the structure
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
            model_points = sim_affine_cloud(affine_set, nmodel,scatter, model_dist, scatter_distro=scatdist)
            back_points = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1 # lognfa
    return nfas*(1/nseeds)

import argparse

def run_experiments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=10,
                    help="path indir  where to find original files")
    ap.add_argument("--npoints", type=int, default=100,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="Cut this number of pixels from each side of image before analysis.")
    ap.add_argument("--detail", type=int, default=40,
                    help="Add this number of pixels to each side of the segmented line / block.")
    args = vars(ap.parse_args())
    nsamp   = args["nsamples"]
    detail  = args["detail"]
    npoints = args["npoints"]
    scatter = args["scatter"]

    scales = np.arange(scatter/10,scatter*4,detail)
    for n in (2,3):
        for m in range(n):
            print(f"n={n} m={m}")
            factors = np.linspace(0.0,(n-m)*0.8,detail)
            distros = [build_scatter_distribution(n - m, f) for f in factors]
            x = np.linspace(0,1,100)
            fbase  = (f'NFA vs scale and decay factor n={n} m={m} s={scatter} N={npoints}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt'):
                nfas = model_vs_scale_and_distro(m, n, distros, scales, nsamp=nsamp,npoints=npoints,scatter=scatter)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', factors)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(factors,'exponential factor',
                                     scales,'analysis scale',
                                     nfas,f'NFA vs decay factor n={n} m={m} s={scatter} N={npoints}')

#==========================================================================================

if __name__ == "__main__":
    print("NFA vs decay factor")
    plt.close('all')
    run_experiments()
