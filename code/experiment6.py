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

from  trotelib import *
from troteplot import *
import matplotlib.cm as cm

def ransac_baseline_test(points,scale,nsamp):
    """
    :return:
    """
    rng = random.default_rng()
    fig = plt.figure(figsize=(14,6))
    #ax = fig.add_subplot()
    ax = plt.subplot(1,2,1)
    ax.scatter(points[:, 0], points[:, 1], alpha=1, s=2)
    plt.title('dataset')
    ax = plt.subplot(1,2,2)
    N,n = points.shape
    m = 1
    candidates = ransac_affine(points,m,nsamp,rng)
    nfas = list()
    cmap = cm.get_cmap("viridis")#LinearSegmentedColormap.from_list("cococho",([1,0,0,1],[.5,.5,.5,.25]))
    nfas= list()
    counts = list()
    for cand in candidates:
        #nfa, count = nfa_ks(points, cand, m, m+1, distance_to_affine, scale, ntests=nsamp, return_counts=True)
        nfa, count = nfa_ks(points, cand, m, m+1, distance_to_affine, scale, ntests=N**2, return_counts=True)
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
    max_nfa = np.max(nfas)
    min_nfa = np.min(nfas)
    print(min_nfa,max_nfa)
    cand_nfas = zip(candidates,nfas)
    det = 0
    for cs in cand_nfas:
        cand,nfa = cs
        color=cmap(nfa/max_nfa)
        if nfa > 0:
            #plot_set(ax, cand, color1=color, color2=color, length=2)
            color = (*color[:3],0.2)
            plot_affine_set_2d_poly(ax, cand, 50, scale, color)
            a_points = np.array(find_aligned_points(points,cand,distance_to_affine,scale))
            plt.scatter(a_points[:, 0], a_points[:, 1], color="gray", s=4, alpha=0.5)
            det += 1
    print('det',det,'not det',len(candidates)-det)
    plt.scatter(a_points[0, 0], a_points[0, 1], alpha=1,s=0.01) # hack para que el colorbar no quede transparente
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

def run_experiment():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=200,
                    help="number of RANSAC samples to draw")
    ap.add_argument("--npoints", type=int, default=1000,
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
    rng = random.default_rng()
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
    bg_points = 30*sim_background_points(npermodel,2, rng)-5
    plt.scatter(bg_points[:,0],bg_points[:,1],color='black',alpha=0.25,s=2)
    fg_points = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=-10, high=10)
        model_cloud = sim_affine_cloud(model, npermodel, rng, scatter=scatter,model_distro=model_distro)
        plt.scatter(model_cloud[:, 0], model_cloud[:, 1],alpha=1,s=2)
        fg_points.append(model_cloud)
    all_points = fg_points
    all_points.append(bg_points)
    all_points = np.concatenate(all_points) # turn list of matrices into one matrix
    fbase  = (f'baseline RANSAC test for a fixed pattern of 3 lines o a plane').lower().replace(' ','_').replace('=','_')
    plt.grid(True)
    plt.show()
    nfas = ransac_baseline_test(all_points,scale=scale,nsamp=nransac)


if __name__ == "__main__":
    print("RANSAC baseline test")
    plt.close('all')
    run_experiment()