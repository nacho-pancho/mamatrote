#!/usr/bin/env python3

import time

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
import os


def parallel_vs_angle(m,n,
                         angles,
                         scales,
                         npoints=100,
                         prop=0.5,
                         scatter_dist=None,
                         bg_dist=None,
                         bg_scale=1,
                         scatter=0.1,
                         seed=42,
                         nsamp=10):
    """
    see wheter we detect the structure or not depending on the effect of a similar structure
    which is oblique to it with a given factor
    """
    rng = random.default_rng(seed)
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    #affine_set_1 = sim_affine_set(n,m,model_dist)
    # we rotate about the z axis, so we need an affine set that is orthogonal to the z axis
    c = np.zeros(n)
    I = np.eye(n)
    V = I[:m,:]
    W = I[m:,:]
    affine_set_1 = (c,V,W)
    nscales = len(scales)
    nang   = len(angles)
    seeds = rng.integers(low=1, high=65535, size=nsamp)
    nseeds = len(seeds)
    nfas = np.zeros((nang,nscales))
    for i,ang in enumerate(angles):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        t0 = time.time()
        # rotate W and last coord of V
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
            affine_set_2 = (c,V2,W2)
        else:
            affine_set_2 = affine_set_1
        for seed in seeds:
            model1_points = sim_affine_cloud(affine_set_1, nmodel, model_dist, scatter_dist, scatter=scatter)
            model2_points = sim_affine_cloud(affine_set_2, nmodel, model_dist, scatter_dist, scatter=scatter)
            back_points  = bg_dist((nback, n))
            model_points = np.concatenate((model1_points,model2_points))
            _test_points = np.concatenate((model_points,back_points))
            for j,s in enumerate(scales):
                nfa = nfa_ks(_test_points, affine_set_1, m, m+1, distance_to_affine, s)
                if seed == seeds[0]:
                    #print(f"\tdist {dist:6} scale {s:6.3f} samples {nsamp:3}  log(nfa) {np.log10(nfa):8.4f}")
                    if ang == angles[nang//2]:
                        plot_two_sets(affine_set_1, affine_set_2, model1_points, model2_points, ran=1)
                        plt.savefig(f"cloud_n_{n}_m{m}_ang_{ang:06.4f}.svg")
                        plt.close()
                nfas[i,j] += nfa < 1 # np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (nang-i)*dt
        print(f'dt={dt:8.2f}s, {rt:8.2f}s to go')
    return  nfas/nseeds


def run_experiments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=10,
                    help="path indir  where to find original files")
    ap.add_argument("--npoints", type=int, default=100,
                    help="text file where input files are specified; each entry should be of the form roll/image.tif")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="Cut this number of pixels from each side of image before analysis.")
    ap.add_argument("--detail", type=int, default=40,
                    help="Add this number of pixels to each side of the segmented line / block.")
    args = vars(ap.parse_args())
    nsamp   = args["nsamples"]
    detail  = args["detail"]
    npoints = args["npoints"]
    scatter = args["scatter"]

    scales = np.linspace(0.01,0.4,detail)#np.logspace(-10,-2,base=2,num=40)
    angles = np.arange(0,np.pi,step=np.pi/40)
    for n in (2,3):
        for m in range(1,n):
            print(f"n={n} m={m}")
            fbase  = (f'NFA vs scale and angle n={n} m={m} s={scatter} N={npoints}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt'):
                nfas = parallel_vs_angle(m, n, angles, scales, nsamp=nsamp,npoints=npoints,scatter=scatter)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', angles)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(angles,'angles',
                                     scales,'analysis scale',
                                     nfas,f'NFA vs angle n={n} m={m} s={scatter} N={npoints}')

#==========================================================================================

import argparse

if __name__ == "__main__":
    print("NFA vs angle between second structure")
    plt.close('all')
    #
    # command line arguments
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxn", type=int, default=1000,
                    help="max number of points to simmulate")
    ap.add_argument("--scatter", type=float, default=0.1,
                    help="Proportion of scale of dispersion from affine set to scale of global point cloud")
    args = vars(ap.parse_args())
    N    = args["maxn"]
    scatter = args["scatter"]
    run_experiments()
