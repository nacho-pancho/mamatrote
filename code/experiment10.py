#!/usr/bin/env python3
"""
Here we combine the greedy version of the RANSAC/NFA algorithm
with a multiscale approach. This version differs from experiment 8 in that it uses
the alternate method (non greedy) for selecting the detected models. See experiment 9
for details.
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


def detect_uniscale(points,scale,nsamp, rng):
    N,n = points.shape
    m = 1
    candidates = ransac_affine(points,m,nsamp,rng)
    cand_points = np.copy(points)
    sig_models = list()
    sig_scores = list()
    sig_points = list()
    for i in range(1000):
        ntests = len(cand_points)
        if ntests <= m+1:
            break
        nfas = [nfa_ks(cand_points, cand, m, m + 1, distance_to_affine, scale, ntests=N**3) for cand in candidates]
        best_idx = np.argmin(nfas)
        best_nfa  = nfas[best_idx]
        #print('model',i,'npoints',len(cand_points),'nfa',best_nfa)
        if best_nfa >= 1: # probando con umbral más exigente
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


def detect_multiscale(points, scale, factor, nsamp, rng, depth=0):
    models, scores, p2 = detect_uniscale(points,scale,nsamp, rng)
    nmodels = len(models)
    print('\t'*depth,'depth',depth,'scale',scale,'points',len(points),'nmodels',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,s,p in zip(models,scores,p2):
            pmat = np.array(p)
            children = detect_multiscale(pmat,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,pmat,children) )
        return model_nodes




import argparse

def run_experiment():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=1000,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=200,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.2,
                    help="How far are the model points scattered from the ground truth element.")
    ap.add_argument("--scale", type=float, default=0.4,
                    help="Analysis scale.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
    ap.add_argument("--recompute", action="store_true",help="Force recomputation even if result exists.")
    args = vars(ap.parse_args())
    nransac = args["nsamples"]
    npoints = args["npoints"]
    scatter = args["scatter"]
    scale   = args["scale"]
    seed    = args["seed"]
    rng = random.default_rng(seed)
    n = 2
    m = 1
    # kind of Anarchy symbol with double horizontal bar
    models = list()
    lp = [[6.5,9],[3,14]]
    models.append(build_affine_set(lp,rng))
    lp = [[5.5,6.5],[1,3]]
    models.append(build_affine_set(lp,rng))
    lp = [[4.5,9],[5,14]]
    models.append(build_affine_set(lp,rng))
    lp = [[3.5,9],[4,14]]
    models.append(build_affine_set(lp,rng))
    k = len(models)
    npermodel = npoints // (k+1)
    bg_points = 30*sim_background_points(npermodel,2,rng)-5
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

    plt.figure(figsize=(8,8))
    plt.scatter(bg_points[:,0],bg_points[:,1],color='black',alpha=0.25,s=2)
    plt.savefig('multiscale_dataset.svg')
    nodes = detect_multiscale(all_points,scale=20,factor=0.6,nsamp=nransac,rng=rng)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    for node in nodes:
        plot_multiscale_ransac_affine(ax, node)
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
    plt.savefig('multiscale_nfa.svg')
    plt.show()
    plt.close()



if __name__ == "__main__":
    print("RANSAC baseline test")
    plt.close('all')
    run_experiment()