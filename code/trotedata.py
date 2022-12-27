"""
TROTELIB
========

Main library for the acronym-free universal detection algorithm
The routines in this module are responsible for creating data points:
   * Estimate models (affine sets) from points
   * Compute distance to affine sets
   * Compute detection scores using various methods
"""
import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import stats
from scipy import special
from trotelib import *


def sim_background_points(npoints, bounding_box, _rng):
    m = len(bounding_box) # bounding box is a list of pairs x[i]_min, x[i]_max where i is the dim
    return tuple(
        tuple(
            _rng.uniform(low=bounding_box[i][0],high=bounding_box[i][1])
            for i in range(m)
        )
        for j in range(npoints)
    )

def sim_affine_set(ambient_dim,affine_dim,distro,rng):
    """
    Simulate an affine set in arbitrary dimension and with arbitrary subdimension
    :param ambient_dim: space where the affine set lives
    :param affine_dim: dimensions of the affine subset
    :return: a pair x_0,(v_1,v_2,...) where x_0 is the offset  and v_1... are the vectors that define the direction of the set
    """
    x0 = distro(ambient_dim)
    V  = distro((affine_dim+1,ambient_dim))
    return build_affine_set(V)


def sim_affine_cloud(_affine_set, _num_points, _rng, scatter = 1.0, model_distro=None, scatter_distro=None):
    """
    given an affine set, simulate a cloud of points such
    that the distribution of their distance to the given affine
    set is distro.
    As most affine sets are infinite (with the exception of a point)
    the sampled dimensions along the set are restricted to (-range,range)
    So, given the dimension m of the affine set, any given point in the
    cloud is simulated as follows:
    1) draw a uniform sample b of size m in the (-range,range) ball
    2) draw a sample c from fdist of size n-m
    3) the simulated point is returned as: x_0 + bV + cW
    :return: num_points simulated points whose distance from the affine set
             is distributed as fdist
    """
    c,V,W,P = _affine_set
    m,n = V.shape
    if model_distro is None:
        #model_distro = lambda x: _rng.uniform(size=x, low=-scatter*10, high=scatter*10)
        model_distro = lambda x: _rng.uniform(size=x, low=0, high=scatter*10)
    if scatter_distro is None:
        scatter_distro = build_scatter_distribution(n - m,_rng)

    n = len(c)
    m = len(V)
    list_of_points = list()
    for i in range(_num_points):
        b = model_distro((m))
        a = _rng.normal(size=(n-m)) # anisotropic
        norm = np.linalg.norm(a)
        d = scatter*scatter_distro(1)
        a *= d/norm
        if len(V) > 0:
            x =  c + b @ V + a @ W
        else:
            x =  c + a @ W
        list_of_points.append(x)
    return list_of_points


def sim_arc_cloud(npoints, model, scatter, rng):
    center,radius,ang1,ang2 = model
    a = rng.uniform(size=npoints,low=ang1,high=ang2)
    r = rng.uniform(size=npoints,low=radius-scatter,high=radius+scatter)
    x = center[0]+r*np.cos(a)
    y = center[1]+r*np.sin(a)

def sim_arc(rng):
    pass

def azucarlito(npoints,scatter,rng):
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
    ground_truth = list()
    npermodel = npoints // (k+1)
    all_points = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=-10, high=10)
        model_points = sim_affine_cloud(model, npermodel, rng, scatter=scatter,model_distro=model_distro)
        ground_truth.append( (model,model_points) )
        all_points.extend(model_points)
    return all_points,ground_truth


def waffle(npoints, scatter, rng, size = 10, nlines=4):
    models = list()
    for i in range(nlines):
        a = (i*size/(nlines-1),0)
        b = (i*size/(nlines-1),size)
        row = build_affine_set((a, b))
        models.append(row)
        print(row)
        a = (0,i*size/(nlines-1))
        b = (size,i*size/(nlines-1))
        col = build_affine_set((a,b))
        models.append( col )
        print(col )
    nmodels = len(models)
    npermodel = npoints//nmodels
    ground_truth = list()
    all_points   = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=0, high=10)
        model_points = sim_affine_cloud(model, npermodel, rng, scatter=scatter, model_distro=model_distro)
        ground_truth.append((model, model_points))
        all_points.extend(model_points)
    return all_points, ground_truth

def satan(npoints, scatter, rng, size = 5, vert = 5, step=2):
    models = list()
    angle = 2.0*np.pi/vert
    i = 0
    while True:
        a = (size*np.cos(i*angle), size*np.sin(i*angle))
        b = (size*np.cos((i+step)*angle), size*np.sin((i+step)*angle))
        print("edge",a,"-",b)
        i = (i + step) % vert
        models.append( build_affine_set((a,b)) )
        if not i:
            break
    nmodels = len(models)
    npermodel = npoints//nmodels
    ground_truth = list()
    all_points   = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=0, high=10)
        model_points = sim_affine_cloud(model, npermodel, rng, scatter=scatter, model_distro=model_distro)
        ground_truth.append((model, model_points))
        all_points.extend(model_points)
    return all_points, ground_truth

def generate_dataset(name,npoints,scatter,rng):
    if name == "azucarlito":
        return azucarlito(npoints, scatter,rng)
    elif name == "waffle":
        return waffle(npoints, scatter,rng)
    elif name == "satan":
        return satan(npoints, scatter,rng)


