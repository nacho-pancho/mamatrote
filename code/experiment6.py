#!/usr/bin/env python3

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
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    N,n = points.shape
    m = 1
    candidates = ransac_affine(points,m,nsamp)
    nfas = list()
    cmap = LinearSegmentedColormap.from_list("cococho",([1,0,0,1],[.5,.5,.5,.25]))
    for cand in candidates:
        #print(cand)
        nfa = nfa_ks(points, cand, m, m+1, distance_to_affine, scale)
        nfas.append(nfa)
        aux = 1 - np.exp(-nfa)
        print(nfa,aux)
        color=cmap(aux)
        plot_set(ax,cand,color1=color,color2=color,length=2)
        if nfa < 1:
            a_points = np.array(find_aligned_points(points,cand,distance_to_affine,scale))
            plt.scatter(a_points[:, 0], a_points[:, 1], color=color, s=1, alpha=0.1)

        #if nfa < 1:
        #    plot_set(ax,cand,color1="red",color2="red",length=2)
        #else:
        #    plot_set(ax, cand, color1=(.5,.5,.5,.25),color2=(.5,.5,.5,.25))
    #print(nfas)
    plt.show()

import argparse

def run_experiment():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=500,
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