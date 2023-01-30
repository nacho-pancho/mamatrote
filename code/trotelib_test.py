#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test various functions of the trotellib (R) acronym-freen detection framework.
"""
import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import matplotlib.cm as colormaps

from trotelib import *
from trotedata import *
from troteplot import *

#==========================================================================================
def test_models_2d():
    N = 100
    n = 2
    m = 1
    rng = random.default_rng(42)
    distro0 = lambda x: rng.uniform(size=x, low=0, high=1)
    affine_set = sim_affine_model(n, m, distro0)
    d1 = lambda x: rng.uniform(size=x, low=0, high=2)
    d2 = lambda x: rng.uniform(size=x, high=0.1)
    fg_points = sim_affine_points(affine_set, N, d1, d2)
    bg_points = rng.uniform(size=(N,2), low=-2,high=2)
    test_points = np.concatenate((fg_points, bg_points))
    x_0, V, W = affine_set
    u = x_0
    v = u + V[0]
    w = u + W[0]
    plt.figure(figsize=(10,10))
    plt.scatter(bg_points[:,0],bg_points[:,1],color='k',alpha=0.2)
    plt.scatter(fg_points[:,0],fg_points[:,1],color='b',alpha=0.2)
    plt.plot((u[0],v[0]),(u[1],v[1]),'r')
    plt.plot((u[0],w[0]),(u[1],w[1]),'g')
    plt.show()


def test_models_3d():
    N = 200
    n = 3
    m = 1
    scatter = 1
    rng = random.default_rng(42)
    distro0 = lambda x: rng.uniform(size=x, low=0, high=1)
    affine_set = sim_affine_model(n, m, distro0)
    d1 = lambda x: rng.uniform(size=x, low=0, high=scatter*10)
    d2 = lambda x: rng.uniform(size=x, high=0.1)
    fg_points = sim_affine_points(affine_set, N, scatter, d1, d2)
    bg_points = rng.uniform(size=(N,n), low=-2,high=2)
    test_points = np.concatenate((fg_points, bg_points))
    x_0, V, W = affine_set
    u = x_0
    v = u + V[0]
    w = u + W[0]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(bg_points[:,0],bg_points[:,1],bg_points[:,2],color='k',alpha=0.1,s=4)
    ax.scatter(fg_points[:,0],fg_points[:,1],fg_points[:,2],color='b',alpha=0.1,s=4)
    ax.plot((u[0],v[0]),(u[1],v[1]),(u[2],v[2]),'r')
    ax.plot((u[0],w[0]),(u[1],w[1]),(u[2],w[2]),'g')
    plt.show()

def test_ks():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 1
    n = 2
    N = 10000
    distro0 = lambda x: rng.uniform(size=x, low=0, high=1)
    affine_set = sim_affine_model(n, m, distro0)
    d1 = lambda x: rng.uniform(size=x, low=-2, high=2)
    d2 = lambda x: rng.laplace(size=x, scale=0.2)
    test_points = sim_affine_points(affine_set, N, d1, d2)
    mm = np.min(test_points)
    MM = np.max(test_points)
    test_points = np.concatenate((test_points, (MM - mm) * random.rand(10 * N, 2) + mm))
    x_0, V, W = affine_set
    a = x_0
    b = x_0 + V[0]
    c = x_0 + W[0]

    scale = 0.5
    affine_set2 = sim_affine_model(n, m, distro0)
    points = np.copy(test_points[:2])
    points += rng.normal(0, 0.01, size=points.shape)
    affine_set3 = build_affine_set(points)
    nfa1 = nfa_ks(test_points, affine_set, m, m + 1, distance_to_affine, scale)
    nfa2 = nfa_ks(test_points, affine_set2, m, m + 1, distance_to_affine, scale)
    nfa3 = nfa_ks(test_points, affine_set3, m, m + 1, distance_to_affine, scale)
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
        scale = 0.025 * s
        nfa = nfa_ks(test_points, affine_set, m, m + 1, distance_to_affine, scale)
        nfas.append(-np.log(nfa))
        scales.append(scale)
    plt.semilogy(scales, nfas)
    plt.xlabel('scale')
    plt.ylabel('logNFA')
    plt.show()

def test_patch_0():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 0
    n = 2
    N = 10000
    a = (5,5)
    cmap = colormaps.get_cmap("hot")
    patch = build_patch([a])
    bounding_box = ((0,10),(0,10))
    x = np.arange(0,10,0.1)
    x,y =  np.meshgrid(x,x)
    points = [(i,j) for i,j in zip(x.ravel(),y.ravel())]
    distances = distance_to_patch(points,patch)
    distances = np.minimum(1,distances)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=cmap(distances),s=16)
    plt.show()

def test_patch_1():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 1
    n = 2
    N = 10000
    a = (3,2)
    b = (3,6)
    cmap = colormaps.get_cmap("hot")
    patch = build_patch([a,b])
    bounding_box = ((0,10),(0,10))
    x = np.arange(0,10,0.1)
    x,y =  np.meshgrid(x,x)
    points = [(i,j) for i,j in zip(x.ravel(),y.ravel())]
    distances = distance_to_patch(points,patch)
    distances = np.minimum(1,distances)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=cmap(distances),s=16)
    plt.show()

def test_patch_2():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 2
    n = 2
    N = 10000
    a = (2,2)
    b = (2,8)
    c = (4,4)
    cmap = colormaps.get_cmap("hot")
    patch = build_patch([a,b,c])
    bounding_box = ((0,10),(0,10))
    x = np.arange(0,10,0.1)
    x,y =  np.meshgrid(x,x)
    points = [(i,j) for i,j in zip(x.ravel(),y.ravel())]
    distances = distance_to_patch(points,patch)
    distances = np.minimum(1,distances)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=cmap(distances),s=16)
    plt.show()

def test_sim_patch_2():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 2
    n = 2
    N = 5000
    a = (2,2)
    b = (2,8)
    c = (4,4)
    #cmap = colormaps.get_cmap("hot")
    patch = build_patch([a,b,c])
    points = sim_patch_points(N, patch, 0.5, rng)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2))
    bbox0 = bounding_box([a,b,c])
    bbox = [(r[0]-2,r[1]+2) for r in bbox0 ]
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_sim_sphere_2d(): #aka circle
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 2
    n = 2
    N = 5000
    c = (4,4)
    r = 3
    sphere = (c,r)
    points = sim_ring_points(N, sphere, 0.5, rng)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2))
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_carucha():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = carucha(1000,rng)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2))
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_collar():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = collar(4000,0.2,rng,big_radius=3,rings=9)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2))
    #plt.xlim(0,10)
    #plt.ylim(0,10)
    plt.show()

def test_some_rings():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = some_rings(4000,rng)
    mat = np.array(points)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2))
    rings = [tmp[0] for tmp in gt]
    for r in rings:
        plot_sphere_2d(ax,r,scatter=0.2)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_clusters_and_lines():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = clusters_and_lines(400,0.2, rng)
    for g in gt:
        print(g[0])
    mat = np.array(points)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2),s=4)
    rings = [tmp[0] for tmp in gt]
    #for r in rings:
    #    plot_sphere_2d(ax,r,scatter=0.2)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_clusters_and_circles():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = clusters_and_circles(1000,0.2, rng)
    mat = np.array(points)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2),s=4)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_lines_and_circles():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = lines_and_circles(1000,0.2, rng)
    mat = np.array(points)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2),s=4)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

def test_clusters_and_lines_and_circles():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    points, gt = clusters_and_lines_and_circles(1000,0.2, rng)
    mat = np.array(points)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    ax.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2),s=4)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()

if __name__ == "__main__":
    #test_models_2d()
    #test_carucha()
    #test_collar()
    test_clusters_and_lines_and_circles()
