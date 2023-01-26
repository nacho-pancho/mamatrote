#!/usr/bin/env python3
"""
This experiment investigates the ability to detect affine structures
as a function of:
a) the analysis scale, which is a parameter of the framework and
b) the total number of data points, which is a property of the input data

The other problem parameters are:
* proportion of model/background points is 50/50.
* scatter distribution is so that the distribution of the
  distance to the affine set is uniform regardless of the dimension
* the scatter distance from a point to the structure defaults to 0.1
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

from trotedata import *
from  trotelib import *
from troteplot import *
import matplotlib.cm as cm

def model_vs_scale_and_npoints(m,n,
                         npointses,
                         scales,
                         rng,
                         prop=0.5,
                         scatter_dist=None,
                         bg_dist=None,
                         bg_scale=1,
                         scatter=0.1,
                         nsamp=10):
    """
    see wheter we detect the structure or not depending on the number of points in it
    :return:
    """
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m,rng)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    affine_set_1 = sim_affine_set(n,m,model_dist,rng)
    nscales = len(scales)
    nnp   = len(npointses)
    nfas = np.zeros((nnp,nscales))
    for i,npoints in enumerate(npointses):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        for k in range(nsamp):
            model_points = sim_affine_cloud(affine_set_1, nmodel, rng, scatter, model_dist, scatter_dist)
            back_points  = bg_dist((nback, n))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set_1, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1
    return  nfas/nsamp


import argparse

if __name__ == "__main__":
    print("affine NFA vs number of points")
    plt.close('all')
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=10,
                    help="path indir  where to find original files")
    ap.add_argument("--npoints", type=int, default=100,
                    help="max. number of points")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="Cut this number of pixels from each side of image before analysis.")
    ap.add_argument("--detail", type=int, default=40,
                    help="Add this number of pixels to each side of the segmented line / block.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    ap.add_argument("--recompute", action="store_true", help="Force recomputation even if result exists.")
    args = vars(ap.parse_args())
    nsamp = args["nsamples"]
    detail = args["detail"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    seed = args["seed"]
    rng = random.default_rng(seed)

    Ns = np.round(np.linspace(max(10, npoints / 10), npoints, detail)).astype(int)
    scales = np.linspace(0.01, 0.4, detail)  # np.logspace(-10,-2,base=2,num=40)
    for n in (2, 3):
        for m in range(n):
            print(f"n={n} m={m}")
            fbase = (f'affine NFA vs scale and npoints n={n} m={m} s={scatter} N={npoints}').lower().replace(' ',
                                                                                                      '_').replace(
                '=', '_')
            print("will perform", len(Ns), "x", len(scales), "tests")
            if not os.path.exists(fbase + '_z.txt') or args["recompute"]:
                nfas = model_vs_scale_and_npoints(m, n, Ns, scales, rng, nsamp=nsamp, scatter=scatter)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', Ns)
            else:
                nfas = np.loadtxt(fbase + '_z.txt')
            ax = plot_scores_img(Ns, 'number of points',
                                 scales, 'analysis scale', nfas,
                                 fbase)
