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


def parallel_vs_distance(m,n,
                         distances,
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
    detect affine line
    :return:
    """
    rng = random.default_rng(seed)
    if scatter_dist is None:
        scatter_dist = build_scatter_distribution(n - m)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x,low=-bg_scale,high=bg_scale)
    model_dist = lambda x: rng.uniform(size=x,low=-bg_scale/2,high=bg_scale/2)

    affine_set_1 = sim_affine_set(n,m,model_dist)
    nscales = len(scales)
    ndist   = len(distances)
    seeds = rng.integers(low=1, high=65535, size=nsamp)
    nseeds = len(seeds)
    nfas = np.zeros((ndist,nscales))
    for i,dist in enumerate(distances):
        nmodel = int(prop*npoints)
        nback  = npoints - nmodel
        t0 = time.time()
        affine_set_2 = build_affine_set_relative_to(affine_set_1, dist=dist, angle=0)
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
                    if dist == distances[-1]:
                        plot_two_sets(affine_set_1, affine_set_2, model1_points, model2_points, ran=1)
                        plt.savefig(f"cloud_n_{n}_m{m}_dist_{dist:06.4f}.svg")
                        plt.close()
                nfas[i,j] += nfa < 1 # np.log(max(nfa,1e-40))
        dt = time.time() - t0
        rt = (ndist-i)*dt
        print(f'dt={dt:8.2f}s, {rt:8.2f}s to go')
    return  nfas/nseeds

def run_experiments():
    nsamp  = 50
    scales = np.arange(0.01,0.41,step=0.01)#np.logspace(-10,-2,base=2,num=40)
    def_scatter = 0.1
    distances = def_scatter*np.arange(0.5,8.5,step=0.125)
    for n in (2,3):
        for m in range(n):
            print(f"\n=======================\nn={n} m={m}")
            print("=======================")
            fbase  = (f'NFA vs scale and distance n={n} m={m}').lower().replace(' ','_').replace('=','_')
            if not os.path.exists(fbase+'_z.txt'):
                nfas = parallel_vs_distance(m, n, distances, scales, nsamp=nsamp)
                np.savetxt(fbase + '_z.txt', nfas)
                np.savetxt(fbase + '_x.txt', scales)
                np.savetxt(fbase + '_y.txt', distances)
            else:
                nfas = np.loadtxt(fbase+'_z.txt')
            ax     = plot_scores_img(distances,'distance',scales,'analysis scale',nfas,f'NFA vs distance n={n} m={m}')

#==========================================================================================


if __name__ == "__main__":
    plt.close('all')
    run_experiments()
