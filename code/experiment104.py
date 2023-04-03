#!/usr/bin/env python3
"""
This experiment investigates the ability to detect an affine structure
when there is another confounding structure at a given angle to the first one.
The results are shown as a function of:
a) the analysis scale, which is a parameter of the framework and
b) the angle between the target and the confounding structure.

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

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from trotedata import *
from trotelib  import *
from troteplot import *
import os


def oblique_vs_angle(m, n,
                     angles,
                     scales,
                     rng,
                     npoints=200,
                     prop=0.75,
                     scatter_dist=None,
                     bg_dist=None,
                     bg_scale=10,
                     scatter=0.1,
                     seed=42,
                     nsamp=10):
    """
    see wheter we detect the structure or not depending on the effect of a similar structure
    which is oblique to it with a given factor
    """
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m, rng)
    bounding_box = tuple((-bg_scale/2, bg_scale/2) for i in range(n))
    # we rotate about the z axis, so we need an affine set that is orthogonal to the z axis
    c = np.zeros(n)
    I = np.eye(n)
    V = I[:m,:]
    W = I[m:,:]
    affine_set_1 = (c,V,W,V)
    nscales = len(scales)
    nang   = len(angles)
    nfas = np.zeros((nang,nscales))
    for i,ang in enumerate(angles):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        # rotate W and last coord of V
        if m == 1 and n == 2 and i == nang//2:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot()
            plot_points(ax, _test_points)
            bbox = fit_bounding_box(_test_points)
            ax.set_xlim(bbox[0][0], bbox[0][1])
            ax.set_ylim(bbox[1][0], bbox[1][1])
            plt.savefig('confounding_oblique.png')
            plt.savefig('confounding_oblique.svg')
            plt.savefig('confounding_oblique.pdf')
            plt.close()
        if ang != 0:
            R      = np.eye(n)
            R[m-1:,m-1] = R[m,m] = np.cos(ang)
            R[m-1,m] = np.sin(ang)
            R[m,m-1] = -np.sin(ang)
            if m > 0:
                V2 = V @ R
            else:
                V2 = V
            W2 = W @ R
            affine_set_2 = (c,V2,W2,V2)
        else:
            affine_set_2 = affine_set_1
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
    ap.add_argument("--proportion", type=float, default=0.5,
                    help="Foreground-background proportion.")
    ap.add_argument("--recompute", action="store_true",help="Force recomputation even if result exists.")
    args = vars(ap.parse_args())
    nsamp   = args["nsamples"]
    detail  = args["detail"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    seed    = args["seed"]
    proportion = args["proportion"]
    rng = random.default_rng(seed)

    scales = np.linspace(0.01,0.4,detail)#np.logspace(-10,-2,base=2,num=40)
    angles = np.arange(0,np.pi,step=np.pi/40)
    for n in (2,3):
        for m in range(1,n):
            print(f"n={n} m={m}")
            fbase  = (f'affine NFA vs scale and angle n={n} m={m} s={scatter} N={npoints}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt') or args["recompute"]:
                nfas = oblique_vs_angle(m, n, angles, scales, rng, prop=proportion, nsamp=nsamp, npoints=npoints, scatter=scatter)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', angles)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(angles,'angles',
                                     scales,'analysis scale',
                                     nfas,fbase)
