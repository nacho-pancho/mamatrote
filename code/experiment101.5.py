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

def model_vs_scale_and_proportion(m,n,
                         proportions,
                         scales,
                         rng,
                         npoints=0.5,
                         scatter_dist=None,
                         bg_scale=10,
                         scatter=0.1,
                         nsamp=10):
    """
    see wheter we detect the structure or not depending on the number of points in it
    :return:
    """
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m,rng)
    bounding_box = tuple((-bg_scale/2, bg_scale/2) for i in range(n))
    affine_set = sim_affine_model(m, bounding_box, rng)
    nscales = len(scales)
    nprop   = len(proportions)
    nfas = np.zeros((nprop,nscales))
    for i,prop in enumerate(proportions):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        for k in range(nsamp):
            model_points = sim_affine_points(affine_set, nmodel, bounding_box, scatter, rng, scatter_dist)
            back_points  = sim_points(nback, bounding_box, rng)
            _test_points = np.concatenate((model_points,back_points))
            # if k == 0 and i == nprop-2:
            #     fig = plt.figure(figsize=(6,6))
            #     ax = fig.gca()
            #     plot_points(ax,_test_points)
            #     bbox = fit_bounding_box(_test_points)
            #     ax.set_xlim(bbox[0][0],bbox[0][1])
            #     ax.set_ylim(bbox[1][0],bbox[1][1])
            #     plt.savefig(fbase+'_sample_dataset.png')
            #     plt.savefig(fbase+'_sample_dataset.svg')
            #     plt.savefig(fbase+'_sample_dataset.pdf')
            #     plt.close()
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set, m, m+1, distance_to_affine, s)
                nfas[i,j] += nfa < 1
    return  nfas/nsamp


import argparse

if __name__ == "__main__":
    print("affine NFA vs proportion of foreground-background points")
    plt.close('all')
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=10,
                    help="path indir  where to find original files")
    ap.add_argument("--npoints", type=int, default=100,
                    help="number of points")
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

    props = np.arange(0.1,1.0,step=0.05)
    scales = np.linspace(0.01, 0.4, detail)  # np.logspace(-10,-2,base=2,num=40)
    for n in (2, 3):
        for m in range(n):
            print(f"n={n} m={m}")
            fbase = (f'affine NFA vs scale and proportion n={n} m={m} s={scatter} N={npoints}').lower().replace(' ',
                                                                                                      '_').replace(
                '=', '_')
            print("will perform", len(props), "x", len(scales), "tests")
            if not os.path.exists(fbase + '_z.txt') or args["recompute"]:
                nfas = model_vs_scale_and_proportion(m, n, props, scales, rng, npoints=npoints, nsamp=nsamp, scatter=scatter)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', props)
            else:
                nfas = np.loadtxt(fbase + '_z.txt')
            ax = plot_scores_img(props, 'proportion',
                                 scales, 'analysis scale', nfas,
                                 fbase)
