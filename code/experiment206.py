#!/usr/bin/env python3
"""
This is the base experiment for the RANSAC-based detection scheme.
Here, instead of testing for all the possible combinations of points in the
dataset that originate different candidate sets, we sample R sets of m+1 the points
(with reposition, where m is the dimension of the patch set) and compute the significance score
on each of them.

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

def ransac_baseline_test(points,scale,nsamp,rng):
    """
    :return:
    """
    fig = plt.figure(figsize=(14,6))
    N   = len(points)
    m = 1
    candidates = ransac_patch(points,m,nsamp,rng)
    print("FUUU")
    print(candidates)
    cmap = cm.get_cmap("viridis")#LinearSegmentedColormap.from_list("cococho",([1,0,0,1],[.5,.5,.5,.25]))
    nfas= list()
    counts = list()
    for cand in candidates:
        nfa, count = nfa_ks(points, cand, m, m+1, distance_to_patch, scale, ntests=N**2, return_counts=True)
        nfas.append(-np.log10(nfa))
        counts.append(count)
    #
    # for debug:
    #
    idx = np.argsort(counts)
    nfas_sorted = np.array(nfas)[idx]
    counts_sorted = np.array(counts)[idx]
    for nfa,cnt in zip(nfas_sorted,counts_sorted):
        print('npoints',cnt,'nfa',nfa)
    #
    # plot stuff
    #
    ax = plt.subplot(1,2,1)
    plot_points(ax,points)
    plt.title('dataset')
    ax = plt.subplot(1,2,2)

    max_nfa = np.max(nfas)
    min_nfa = np.min(nfas)
    print(min_nfa,max_nfa)
    cand_nfas = zip(candidates,nfas)
    det = 0
    for cs in cand_nfas:
        cand,nfa = cs
        color=cmap(nfa/max_nfa)
        if nfa > 0:
            color = (*color[:3],0.2)
            plot_patch_set_2d_poly(ax, cand, 50, scale, color)
            a_points = np.array(find_aligned_points(points,cand,distance_to_patch,scale))
            plt.scatter(a_points[:, 0], a_points[:, 1], color="gray", s=4, alpha=0.5)
            det += 1
    print('det',det,'not det',len(candidates)-det)

    plt.colorbar()
    xmin = np.min([p[0] for p in points])
    xmax = np.max([p[0] for p in points])
    ymin = np.min([p[1] for p in points])
    ymax = np.max([p[1] for p in points])
    xlen = xmax-xmin
    ylen = ymax-ymin
    maxlen = max(xlen,ylen)
    plt.xlim(xmin,xmin+maxlen)
    plt.ylim(ymin,ymin+maxlen)
    plt.title('detected models')
    plt.show()


import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=200,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=1000,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.2,
                    help="How far are the model points scattered from the ground truth element.")
    ap.add_argument("--scale", type=float, default=0.4,
                    help="Analysis scale.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    ap.add_argument("--recompute", action="store_true", help="Force recomputation even if result exists.")
    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    scale = args["scale"]
    seed = args["seed"]
    rng = random.default_rng(seed)
    all_points, ground_truth = azucarlito(npoints, scatter,rng)
    nfas = ransac_baseline_test(all_points, scale, nransac, rng)

    fbase = (f'baseline RANSAC test for a fixed pattern of 4 lines o a plane').lower().replace(' ', '_').replace('=',
                                                                                                                 '_')