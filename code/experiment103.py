#!/usr/bin/env python3
"""
This experiment investigates the ability to detect an affine structure
when there is another confounding structure, parallel to the first one.
The results are shown as a function of:
a) the analysis scale, which is a parameter of the framework and
b) the distance between the target and the confounding structure.

The other problem parameters are:

* proportion of model/background points is 50/50.
* number of points defaults to 100
  distance to the affine set is uniform regardless of the dimension
* the experiment is repeated 10 times for 10 different random seeds
* scatter distribution is so that the distribution of the
  distance to the affine set is uniform regardless of the dimension
* the scatter distance from a point to the structure defaults to 0.1

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

from trotedata import *
from  trotelib import *
from troteplot import *


def parallel_vs_distance(m,n,
                         distances,
                         scales,
                         rng,
                         npoints=100,
                         prop=0.5,
                         scatter_dist=None,
                         bg_dist=None,
                         bg_scale=1,
                         scatter=0.1,
                         nsamp=10):
    """
    see wheter we detect the structure when another similar structure is parallel to it
    at a given distance.
    :return:
    """
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m, rng)
    bounding_box = tuple((-bg_scale/2, bg_scale/2) for i in range(n))
    affine_set_1 = sim_affine_model(m, bounding_box, rng)
    nscales = len(scales)
    ndist   = len(distances)
    nfas = np.zeros((ndist,nscales))
    for i,dist in enumerate(distances):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        affine_set_2 = build_affine_set_relative_to(affine_set_1, dist=dist, angle=0)
        for k in range(nsamp):
            back_points  = sim_points(nback, bounding_box, rng)
            model1_points = sim_affine_points(affine_set_1, nmodel, bounding_box, scatter, rng, scatter_dist)
            model2_points = sim_affine_points(affine_set_2, nmodel, bounding_box, scatter, rng, scatter_dist)
            model_points = np.concatenate((model1_points,model2_points))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set_1, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1
    return  nfas/nsamp

#==========================================================================================

import argparse

if __name__ == "__main__":
    print("affine NFA vs distance between second structure")
    plt.close('all')
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=10,
                    help="path indir  where to find original files")
    ap.add_argument("--npoints", type=int, default=100,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="Cut this number of pixels from each side of image before analysis.")
    ap.add_argument("--detail", type=int, default=40,
                    help="Add this number of pixels to each side of the segmented line / block.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    ap.add_argument("--recompute", action="store_true",help="Force recomputation even if result exists.")
    args = vars(ap.parse_args())
    nsamp   = args["nsamples"]
    detail  = args["detail"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    seed    = args["seed"]
    rng = random.default_rng(seed)

    scales = np.linspace(0.01,0.4,detail)#np.logspace(-10,-2,base=2,num=40)
    distances = scatter*np.linspace(0.5,8,detail)
    for n in (2,3):
        for m in range(n):
            print(f"n={n} m={m}")
            fbase  = (f'affine NFA vs scale and distance n={n} m={m} s={scatter} N={npoints}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt') or args["recompute"]:
                nfas = parallel_vs_distance(m, n, distances, scales, rng, nsamp=nsamp, scatter=scatter,npoints=npoints)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', distances)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(distances,'distance',
                                     scales,'analysis scale',
                                     nfas,fbase)
