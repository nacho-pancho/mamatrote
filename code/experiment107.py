#!/usr/bin/env python3
"""
Here we test a greedy version of the RANSAC/NFA algorithm.
Given a set of points, and a set of candidates, we find the most significant model.
We save it and remove all its nearby points from the dataset.
We repeat this until there are no new significant sets.
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
from trotelib import *
from troteplot import *
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=500,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=500,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--nbpoints", type=int, default=100,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="How far are the model points scattered from the ground truth element.")
    ap.add_argument("--scale", type=float, default=0.3,
                    help="Analysis scale.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    ap.add_argument("--recompute", action="store_true",help="Force recomputation even if result exists.")
    ap.add_argument("--dataset",type=str, default="azucarlito", help="which dataset to test.")
    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    scale   = args["scale"]
    seed    = args["seed"]
    dataset = args["dataset"]
    rng = random.default_rng(seed)
    #
    all_points, ground_truth = generate_dataset(dataset, npoints, scatter, rng)
    bbox = fit_bounding_box(all_points)
    bg_points = sim_background_points(args["nbpoints"],bbox,rng)
    all_points.extend(bg_points)
    ground_truth.append(("background",bg_points))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    plot_points(ax,all_points)

    models = ransac_nfa_affine_uniscale_greedy(all_points,scale,nransac,rng)
    scores = [-np.log10(m[2]) for m in models]
    model_points = [m[1] for m in models]
    params = [ m[0] for m in models]
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    plot_points(ax,all_points,alpha=1,size=2)
    plt.savefig(f'uniscale_nfa_{dataset}_greedy_data.png')
    plt.savefig(f'uniscale_nfa_{dataset}_greedy_data.svg')
    plt.savefig(f'uniscale_nfa_{dataset}_greedy_data.pdf')
    plt.close()

    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    plot_uniscale_ransac_affine(ax, all_points, params, scores, model_points, scale)
    plt.savefig(f'uniscale_nfa_{dataset}_greedy_result.png')
    plt.savefig(f'uniscale_nfa_{dataset}_greedy_result.svg')
    plt.savefig(f'uniscale_nfa_{dataset}_greedy_result.pdf')
    plt.close()
