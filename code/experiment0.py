#!/usr/bin/env python3

import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import stats
from scipy import special
from  trotelib import *



#==========================================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 1
    n = 2
    N = 10000
    distro0 = lambda x: rng.uniform(size=x, low=0, high=1)
    affine_set  = sim_affine_set(n,m,distro0)
    d1 = lambda x: rng.uniform(size=x,low=-2,high=2)
    d2 = lambda x: rng.laplace(size=x,scale=0.2)
    test_points = sim_affine_cloud(affine_set, N, d1,d2)
    mm = np.min(test_points)
    MM = np.max(test_points)
    test_points = np.concatenate((test_points, (MM-mm)*random.rand(10*N, 2)+mm))
    x_0, V, W = affine_set
    a = x_0
    b = x_0 + V[0]
    c = x_0 + W[0]

    scale = 0.5
    affine_set2 = sim_affine_set(n, m, distro0)
    points = np.copy(test_points[:2])
    points += rng.normal(0, 0.01, size=points.shape)
    affine_set3 = build_affine_set(points)
    nfa1 = nfa_ks(test_points, affine_set, m, m+1, distance_to_affine, scale)
    nfa2 = nfa_ks(test_points, affine_set2, m, m+1, distance_to_affine, scale)
    nfa3 = nfa_ks(test_points, affine_set3, m, m+1, distance_to_affine, scale)
    print(f'Scale {scale}')
    print(f'Red NFA {nfa1}')
    print(f'Green NFA {nfa2}')
    print(f'Blue NFA {nfa3}')
    x_0, V, W = affine_set
    a1 = x_0
    b1 = x_0 + V[0]
    c1 = x_0 + W[0]
    x_0, V, W = affine_set2
    a2 = x_0
    b2 = x_0 + V[0]
    c2 = x_0 + W[0]
    x_0, V, W = affine_set3
    a3 = x_0
    b3 = x_0 + V[0]
    c3 = x_0 + W[0]
    nfas = []
    scales = []
    for s in range(20):
        scale = 0.025*s
        nfa = nfa_ks(test_points, affine_set, m, m+1, distance_to_affine, scale)
        nfas.append(-np.log(nfa))
        scales.append(scale)
    plt.semilogy(scales, nfas)
    plt.xlabel('scale')
    plt.ylabel('logNFA')
    plt.show()
