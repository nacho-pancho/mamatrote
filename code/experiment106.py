#!/usr/bin/env python3
"""
This is the base experiment for the RANSAC-based detection scheme.
Here, instead of testing for all the possible combinations of points in the
dataset that originate different candidate sets, we sample R sets of m+1 the points
(with reposition, where m is the dimension of the affine set) and compute the significance score
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

def ransac_baseline_test(points,scale,nsamp,fbase,rng):
    """
    :return:
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    plot_points(ax,points)
    plt.savefig(fbase + '_data_points.pdf')
    plt.savefig(fbase + '_data_points.png')
    plt.savefig(fbase + '_data_points.svg')
    plt.close(fig)
    bbox = fit_bounding_box(points)
    fig = plt.figure(figsize=(6,6))
    N   = len(points)
    m = 1
    candidates = ransac_affine(points,m,nsamp,rng)
    print('candidates:',len(candidates))
    cmap = cm.get_cmap("viridis")
    nfas= list()
    counts = list()
    for i,cand in enumerate(candidates):
        nfa, count = nfa_ks(points, cand, m, m+1, distance_to_affine, scale, ntests=N**2, return_counts=True)
        nfas.append(-np.log10(nfa))
        counts.append(count)
        if i < 15:
            plot_affine_model_2d(ax, cand, 50, scale, color=(0,0.25,0.5,0.1))
            a_points = np.array(find_aligned_points(points, cand, distance_to_affine, scale))
            plt.scatter(a_points[:, 0], a_points[:, 1], color="black", s=3, alpha=0.5)
    plt.xlim(bbox[0][0],bbox[0][1])
    plt.ylim(bbox[1][0],bbox[1][1])
    plt.savefig(fbase + '_some_candidates.pdf')
    plt.savefig(fbase + '_some_candidates.png')
    plt.savefig(fbase + '_some_candidates.svg')
    plt.close(fig)

    raw_models = detect(points,candidates,m,m+1,distance_to_affine,scale)
    print('raw models:',len(raw_models))
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    for i,cand in enumerate(raw_models):
        cand_params, cand_points, cand_nfa = cand
        plot_affine_model_2d(ax, cand_params, 50, scale, color=(0,0.25,0.5,0.1))
        a_points = np.array(cand_points)
        plt.scatter(a_points[:, 0], a_points[:, 1], color="black", s=3, alpha=0.5)
    plt.xlim(bbox[0][0],bbox[0][1])
    plt.ylim(bbox[1][0],bbox[1][1])
    plt.savefig(fbase + '_raw_candidates.pdf')
    plt.savefig(fbase + '_raw_candidates.png')
    plt.savefig(fbase + '_raw_candidates.svg')
    plt.close(fig)

    filtered_models = mask_greedy(points, raw_models, m, m+1, distance_to_affine, scale, debug=True)
    print('filtered models:',len(filtered_models))
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    for i,cand in enumerate(filtered_models):
        cand_param, cand_points, cand_nfas = cand
        plot_affine_model_2d(ax, cand_param, 50, scale, color=(0,0.25,0.5,0.1))
        a_points = np.array(cand_points)
        plt.scatter(a_points[:, 0], a_points[:, 1], color="black", s=3, alpha=0.5)
    plt.xlim(bbox[0][0],bbox[0][1])
    plt.ylim(bbox[1][0],bbox[1][1])
    plt.savefig(fbase + '_filtered_candidates.pdf')
    plt.savefig(fbase + '_filtered_candidates.png')
    plt.savefig(fbase + '_filtered_candidates.svg')
    plt.close(fig)

    #
    # for debug:
    #
    #idx = np.argsort(counts)
    #nfas_sorted = np.array(nfas)[idx]
    #counts_sorted = np.array(counts)[idx]
    #for nfa,cnt in zip(nfas_sorted,counts_sorted):
    #    print('npoints',cnt,'nfa',nfa)
    #
    # plot stuff
    #
    fig = plt.figure(figsize=(14,6))
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
            plot_affine_model_2d(ax, cand, 50, scale, color)
            a_points = np.array(find_aligned_points(points,cand,distance_to_affine,scale))
            plt.scatter(a_points[:, 0], a_points[:, 1], color="gray", s=4, alpha=0.5)
            det += 1
    print('det',det,'not det',len(candidates)-det)
    plt.colorbar()
    plt.xlim(bbox[0][0],bbox[0][1])
    plt.ylim(bbox[1][0],bbox[1][1])
    plt.title('detected models')
    plt.savefig('experiment106.svg')
    plt.close()

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
    ap.add_argument("--dataset",type=str, default="azucarlito", help="which dataset to test.")
    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    dataset = args["dataset"]
    scale = args["scale"]
    seed = args["seed"]
    rng = random.default_rng(seed)
    #all_points, ground_truth = azucarlito(npoints, scatter,rng)
    #nfas = ransac_baseline_test(all_points, scale, nransac, rng)
    all_points, ground_truth = generate_dataset(dataset, npoints, scatter, rng)
    bbox = fit_bounding_box(all_points)
    bg_points = sim_background_points(npoints//2,bbox,rng)
    all_points.extend(bg_points)
    ground_truth.append(("background",bg_points))
    fbase = (f'affine {dataset} ransac').lower().replace(' ', '_').replace('=','_')
    nfas = ransac_baseline_test(all_points, scale, nransac, fbase, rng)
