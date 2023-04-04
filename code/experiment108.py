#!/usr/bin/env python3
"""
Here we combine the greedy version of the RANSAC/NFA algorithm
with a multiscale approach. This generates a hierarchy of models where
the topmost model(s) (may be more than 1) correspond to the coarsest (larger) scale
and then the children contain submodels found within their parent points when
analyzed at a smaller scale. This goes on until no significant models are found at a given scale.
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
from trotelib  import *
from troteplot import *
import argparse
import cProfile

if __name__ == "__main__":
    print("RANSAC baseline test")
    plt.close('all')
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=200,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=200,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--nbpoints", type=int, default=50,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.2,
                    help="How far are the model points scattered from the ground truth element.")
    ap.add_argument("--scale", type=float, default=0.2,
                    help="Analysis scale.")
    ap.add_argument("--factor", type=float, default=0.5,
                    help="reduce scale at each level.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    ap.add_argument("--dataset",type=str, default="azucarlito", help="which dataset to test.")
    ap.add_argument("--recompute", action="store_true",help="Force recomputation even if result exists.")
    ap.add_argument("--no-plot-branches", action="store_true",help="Do not plot model tree branches.")
    ap.add_argument("--no-plot-leaves",  action="store_true", help="Do not plot leaves.")
    ap.add_argument("--no-plot-single",  action="store_true", help="Do not plot single parents.")

    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    scale   = args["scale"]
    seed    = args["seed"]
    factor  = args["factor"]
    dataset = args["dataset"]
    rng = random.default_rng(seed)
    all_points, ground_truth = generate_dataset(dataset, npoints, scatter, rng)

    bbox = fit_bounding_box(all_points)
    bg_points = sim_background_points(args["nbpoints"],bbox,rng)
    all_points.extend(bg_points)
    ground_truth.append(("background",bg_points))

    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    plot_points(ax,all_points,alpha=1,size=2)
    plt.savefig(f'multiscale_nfa_{dataset}_greedy_data.png')
    plt.savefig(f'multiscale_nfa_{dataset}_greedy_data.svg')
    plt.savefig(f'multiscale_nfa_{dataset}_greedy_data.pdf')
    plt.close()

    nodes = ransac_nfa_affine_multiscale_greedy(all_points,scale=20,factor=factor,nsamp=nransac,rng=rng)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    for node in nodes:
        plot_multiscale_ransac_affine(ax, node,
                                      not args["no_plot_leaves"],
                                      not args["no_plot_single"],
                                      not args["no_plot_branches"])
    xmin = 0.9*np.min([p[0] for p in all_points])
    xmax = 1.1*np.max([p[0] for p in all_points])
    ymin = 0.9*np.min([p[1] for p in all_points])
    ymax = 1.1*np.max([p[1] for p in all_points])
    xlen = (xmax-xmin)
    ylen = (ymax-ymin)
    maxlen = max(xlen,ylen)
    plt.xlim(xmin,xmin+maxlen)
    plt.ylim(ymin,ymin+maxlen)
    plt.title('detected models')
    plt.savefig(f'multiscale_nfa_{dataset}_greedy_result.png')
    plt.savefig(f'multiscale_nfa_{dataset}_greedy_result.svg')
    plt.savefig(f'multiscale_nfa_{dataset}_greedy_result.pdf')
    plt.close()



