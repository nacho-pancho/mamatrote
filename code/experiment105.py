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
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from trotedata import *
from  trotelib import *
from troteplot import *

import matplotlib.cm as cm




def model_vs_scale_and_distro(m,n,
                              scatter_distros,
                              scales,
                              rng,
                              scatter=0.1,
                              bg_dist=None,
                              bg_scale=1,
                              prop=0.5,
                              nsamp=10,
                              npoints=100):
    """
    see wheter we detect the structure or not depending on how concentrated the points are around the structure
    :return:
    """
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    bounding_box = tuple((-bg_scale/2, bg_scale/2) for i in range(n))
    affine_set = sim_affine_model(m, bounding_box, rng)
    ndistros = len(scatter_distros)
    nscales  = len(scales)
    nfas = np.zeros((ndistros,nscales))
    for i,scatdist in enumerate(scatter_distros):
        nmodel = int(np.ceil(prop*npoints))
        nback  = npoints - nmodel
        for k in range(nsamp):
            model_points = sim_affine_points(affine_set, nmodel, bounding_box, scatter, rng, scatter_distro=scatdist)
            back_points  = sim_points(nback, bounding_box, rng)
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1
    return nfas*(1/nsamp)



#==========================================================================================

import argparse

if __name__ == "__main__":
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
    scales = np.linspace(scatter/10,scatter*4,detail)
    for n in (2,3):
        for m in range(n):
            print(f"n={n} m={m}")
            factors = np.linspace(0.0,(n-m)*0.8,detail)
            distros = [build_scatter_distribution(n - m, rng, f) for f in factors]
            print("will perform ",len(factors),"x",len(scales),"tests")
            x = np.linspace(0,1,100)
            fbase  = (f'affine NFA vs scale and decay factor n={n} m={m} s={scatter} N={npoints}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt') or args["recompute"]:
                nfas = model_vs_scale_and_distro(m, n, distros, scales, rng, nsamp=nsamp, npoints=npoints, scatter=scatter)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', factors)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(factors,'exponential factor',
                                     scales,'analysis scale',
                                     nfas,fbase)
