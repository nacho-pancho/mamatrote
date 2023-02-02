#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from trotedata import *
from trotelib import *
from troteplot import *
from matplotlib import cm as colormaps

def test_patch_0():
    import matplotlib.pyplot as plt
    rng = random.default_rng()
    m = 0
    n = 2
    N = 10000
    a = (5,5)
    cmap = colormaps.get_cmap("hot")
    patch = fit_patch_model([a])
    bounding_box = ((0,10),(0,10))
    x = np.arange(0,10,0.1)
    x,y =  np.meshgrid(x,x)
    points = [(i,j) for i,j in zip(x.ravel(),y.ravel())]
    distances = patch.distance(points)
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
    patch = fit_patch_model([a, b])
    bounding_box = ((0,10),(0,10))
    x = np.arange(0,10,0.1)
    x,y =  np.meshgrid(x,x)
    points = [(i,j) for i,j in zip(x.ravel(),y.ravel())]
    distances = patch.distance(points)
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
    patch = fit_patch_model([a, b, c])
    bounding_box = ((0,10),(0,10))
    x = np.arange(0,10,0.1)
    x,y =  np.meshgrid(x,x)
    points = [(i,j) for i,j in zip(x.ravel(),y.ravel())]
    distances = patch.distance(points)
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
    patch = fit_patch_model([a, b, c])
    points = sim_patch_points(N, patch, 0.5, rng)
    mat = np.array(points)
    plt.figure(figsize=(10,10))
    plt.scatter(mat[:,0],mat[:,1],c=(0,0,0,0.2))
    bbox0 = fit_bounding_box([a, b, c])
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
    points = sim_sphere_points(N, sphere, 0.5, rng)
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
        plot_sphere_model_2d(ax, r, scatter=0.2)
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
