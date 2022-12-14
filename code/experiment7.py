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

from  trotelib import *
from troteplot import *
import matplotlib.cm as cm


def detect_uniscale(points,scale,nsamp):
    N,n = points.shape
    m = 1
    candidates = ransac_affine(points,m,nsamp)
    cand_points = np.copy(points)
    sig_models = list()
    sig_scores = list()
    sig_points = list()
    for i in range(1000):
        ntests = len(cand_points)
        nfas = [nfa_ks(cand_points, cand, m, m + 1, distance_to_affine, scale, ntests=ntests) for cand in candidates]
        best_idx = np.argmin(nfas)
        best_nfa  = nfas[best_idx]
        print(i,len(cand_points),best_nfa)
        if best_nfa >= 0.1: # probando con umbral más exigente
            break
        best_cand = candidates[best_idx]
        best_points = find_aligned_points(cand_points,best_cand,distance_to_affine,scale)
        sig_models.append(best_cand)
        sig_scores.append(best_nfa)
        sig_points.append(best_points)
        #
        # remove aligned points
        # the easiest way is to use set operations
        #
        aux1 = [tuple(c) for c in cand_points]
        aux2 = [tuple(c) for c in best_points]
        cand_points = np.array(list(set(aux1).difference(set(aux2))))
    return sig_models,sig_scores,sig_points



import argparse

def run_experiment():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=500,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=200,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.2,
                    help="How far are the model points scattered from the ground truth element.")
    ap.add_argument("--scale", type=float, default=0.4,
                    help="Analysis scale.")
    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    scale   = args["scale"]
    #n       = args["ambient_dim"]
    #m       = args["affine_dim"]
    #k       = args["nstruct"]
    #
    n = 2
    m = 1
    # kind of Anarchy symbol with double horizontal bar
    models = list()
    lp = [[6.5,9],[3,14]]
    models.append(build_affine_set(lp))
    lp = [[5.5,6.5],[1,3]]
    models.append(build_affine_set(lp))
    lp = [[4.5,9],[5,14]]
    models.append(build_affine_set(lp))
    lp = [[3.5,9],[4,14]]
    models.append(build_affine_set(lp))
    k = len(models)
    npermodel = npoints // (k+1)
    plt.figure(figsize=(8,8))
    bg_points = 30*sim_background_points(npermodel,2)-5
    plt.scatter(bg_points[:,0],bg_points[:,1],color='black',alpha=0.25,s=2)
    fg_points = list()
    for model in models:
        rng = random.default_rng(seed=42)
        model_distro = lambda x: rng.uniform(size=x, low=-10, high=10)
        model_cloud = sim_affine_cloud(model, npermodel, scatter=scatter,model_distro=model_distro)
        plt.scatter(model_cloud[:, 0], model_cloud[:, 1],alpha=1,s=4)
        fg_points.append(model_cloud)
    all_points = fg_points
    all_points.append(bg_points)
    all_points = np.concatenate(all_points) # turn list of matrices into one matrix
    fbase  = (f'baseline RANSAC test for a fixed pattern of 3 lines o a plane').lower().replace(' ','_').replace('=','_')
    plt.grid(True)
    plt.show()
    models,scores,model_points = detect_uniscale(all_points,scale=scale,nsamp=nransac)
    scores = [-np.log10(s) for s in scores]
    plot_uniscale_ransac_affine(all_points, models, scores, model_points, scale)


if __name__ == "__main__":
    print("RANSAC baseline test")
    plt.close('all')
    run_experiment()