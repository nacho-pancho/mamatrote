#!/usr/bin/env python3
"""
Here we test a greedy version of the RANSAC/NFA algorithm.
Given a set of points, and a set of candidates, we find the most significant model.
This experiment differs from experiment 7 in the way that overlappings (redundant) models
are removed.

In experiment 7, we select the most significant model, remove its points, and do
everything again (including the NFAs) until there are no significant models.

In this case, we do the NFAs once and use them to rank the models.
As before, we begin with the most significant model.

Now, instead of removing the points and re-computing everything,
we tentatively remove the points from the remaining candidates and check
for their significance without those points.

Then, the models that are no longer significant after removing the aforementioned points are removed.
Then we move down the list of the reminding candidates, in significance order, and do the same
with less significant models.

And so on until the end of the list.
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
    rng = random.default_rng()
    candidates = ransac_affine(points,m,nsamp,rng)
    cand_models = list(candidates)
    sig_models = list()
    sig_scores = list()
    sig_points = list()
    #
    # The baseline (global ) NFAs are computed _once_
    #
    nfas = [nfa_ks(points, cand, m, m + 1, distance_to_affine, scale, ntests=N ** 3) for cand in candidates]
    #
    # the candidates are sorted _once_ using their NFAs
    # the most significant have lower NFA, so the ascending order is ok
    #
    idx = np.argsort(nfas)
    nfas_sorted = [nfas[i] for i in idx]
    nfas = nfas_sorted
    cand_models_sorted = [cand_models[i] for i in idx]
    cand_models = cand_models
    cand_points = [find_aligned_points(points, c, distance_to_affine, scale) for c in cand_models]
    print(nfas)
    cand_models = list(zip(cand_models,nfas,cand_points))
    # now we repeat
    #
    more_cand = True
    top = 0
    while len(cand_models):
        best_cand,best_nfa,best_points   = cand_models[0]
        print("NFA of best model",best_nfa)
        if best_nfa >= 1:
            break
        sig_models.append((best_cand,best_nfa,best_points))
        filtered_models = list()
        for t in range(1,len(cand_models)):
            other_model,other_nfa,other_points = cand_models[t]
            #
            # remove the best candidate points from this model
            #
            best_aux  = [tuple(c) for c in best_points]
            other_aux = [tuple(c) for c in other_points]
            other_rem = np.array(list(set(other_aux).difference(set(best_aux))))
            #print(f"{t:5} other points {len(other_points):6} other non-redundant points {len(other_rem):6}",end=" ")
            if len(other_rem) <= m+1:
                continue
                print(" not enough points",end=" ")
            #
            # see if it is still significant
            #
            rem_nfa = nfa_ks(other_rem, other_model, m, m + 1, distance_to_affine, scale, ntests=N ** 2)
            #print(f"orig NFA {other_nfa:16.4f} NFA of non-redundant points {rem_nfa:16.4f}",end=" ")
            #
            # if it is, it is the new top
            #
            if rem_nfa < 1:
            #    print("KEPT!")
                filtered_models.append((other_model,other_nfa,other_points))
            else:
                pass#print("REMOVED!")
            # if other_nfa >= 1, the top is incremented but the other model is _not_ added to the list
        #
        #
        # we continue the analysis with the filtered models
        print("redundant ",len(cand_models)-len(filtered_models),"non-redundant ",len(filtered_models))
        cand_models = filtered_models
    print("kept ", len(sig_models))
    return sig_models



import argparse

def run_experiment():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=500,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=200,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="How far are the model points scattered from the ground truth element.")
    ap.add_argument("--scale", type=float, default=0.4,
                    help="Analysis scale.")
    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    scale   = args["scale"]
    rng = random.default_rng()
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
    bg_points = 30*sim_background_points(npermodel,2,rng)-5
    plt.scatter(bg_points[:,0],bg_points[:,1],color='black',alpha=0.25,s=2)
    fg_points = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=-10, high=10)
        model_cloud = sim_affine_cloud(model, npermodel, rng, scatter=scatter,model_distro=model_distro)
        plt.scatter(model_cloud[:, 0], model_cloud[:, 1],alpha=1,s=4)
        fg_points.append(model_cloud)
    all_points = fg_points
    all_points.append(bg_points)
    all_points = np.concatenate(all_points) # turn list of matrices into one matrix
    fbase  = (f'baseline RANSAC test for a fixed pattern of 3 lines o a plane').lower().replace(' ','_').replace('=','_')
    plt.savefig('uniscale_dataset.svg')
    #plt.close()
    models = detect_uniscale(all_points,scale=scale,nsamp=nransac)
    model_scores = [-np.log10(s[1]) for s in models]
    model_models = [s[0] for s in models]
    model_points = [s[2] for s in models]
    for p in model_points:
        print(len(p))
    fig = plt.figure(figsize=(14,6))
    ax = plt.subplot(1,2,1)
    ax.scatter(all_points[:, 0], all_points[:, 1], alpha=1, s=2)
    plt.title('dataset')

    ax = plt.subplot(1,2,2)
    #
    # "unzip"
    #

    plot_uniscale_ransac_affine(ax, all_points, model_models, model_scores, model_points, scale)

    plt.savefig('uniscale_nfa.svg')
    #plt.close()
    plt.show()

if __name__ == "__main__":
    print("RANSAC baseline test")
    plt.close('all')
    run_experiment()