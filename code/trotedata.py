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

def sim_patch_cloud(_num_points, _patch, _scatter, _rng):
    """
    simulate a cloud of points whose distribution is uniform within the region defined
    by all points whose distance to the patch is less than a scatter value.
    We sample values uniformly from a bounding box and keep those inside
    """
    c,V,W,P = _patch
    bbox0  = bounding_box(P)
    bbox = [(r[0]-_scatter,r[1]+_scatter) for r in bbox0 ]
    n = len(c)
    m = len(V)
    _sim_points = list()
    _rem_points = _num_points
    while _rem_points > 0:
        _raw_points = sim_background_points(_rem_points*2,bbox,_rng)
        _distances  = distance_to_patch(_raw_points,_patch)
        _inner_points = [p for p,d in zip (_raw_points,_distances) if d <= _scatter]
        _sim_points.extend(_inner_points)
        _rem_points -= len(_inner_points)
    if len(_sim_points) > _num_points:
        _sim_points = _sim_points[:_num_points]
    return _sim_points

def sim_sphere_cloud(_num_points, _sphere, _scatter, _rng):
    """
    simulate a cloud of points whose distribution is uniform within the region defined
    by all points whose distance to the patch is less than a scatter value.
    We sample values uniformly from a bounding box and keep those inside
    """
    c,r = _sphere
    c = np.array(c)
    n = len(c)
    baux1 = c-r-_scatter
    baux2 = c+r+_scatter
    bbox = [(baux1[i],baux2[i]) for i in range(n) ]
    _sim_points = list()
    _rem_points = _num_points
    while _rem_points > 0:
        _raw_points = sim_background_points(_rem_points*2,bbox,_rng)
        _distances  = distance_to_sphere(_raw_points,_sphere)
        _inner_points = [p for p,d in zip (_raw_points,_distances) if d <= _scatter]
        _sim_points.extend(_inner_points)
        _rem_points -= len(_inner_points)
    if len(_sim_points) > _num_points:
        _sim_points = _sim_points[:_num_points]
    return _sim_points


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
        a = (0,i*size/(nlines-1))
        b = (size,i*size/(nlines-1))
        col = build_affine_set((a,b))
        models.append( col )
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

def some_rings(npoints,rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    npermodel = npoints // 5
    # ojo
    model = ((2,6),1.5)
    model_points = sim_sphere_cloud(npermodel, model, 0.1, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # pupila
    model = ((2,6),0.5)
    model_points = sim_sphere_cloud(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # otro ojo
    model = ((6,6),1)
    model_points = sim_sphere_cloud(npermodel, model, 0.1, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth

def carucha(npoints,rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    npermodel = npoints // 5
    # cara
    model = ((5,5),5)
    model_points = sim_sphere_cloud(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # nariz
    model = ((5,4),1)
    model_points = sim_sphere_cloud(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # ojo
    model = ((3,6),2.2)
    model_points = sim_sphere_cloud(npermodel, model, 0.1, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # pupila
    model = ((3.2,6),0.5)
    model_points = sim_sphere_cloud(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # otro ojo
    model = ((7,6),2)
    model_points = sim_sphere_cloud(npermodel, model, 0.1, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # pupila
    model = ((7,6),0.7)
    model_points = sim_sphere_cloud(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth

def collar(npoints, scatter, rng, big_radius = 5, small_radius = 1, rings = 7):
    models = list()
    angle = 2.0*np.pi/rings
    for i in range(rings):
        c = (big_radius * np.cos(i*angle), big_radius * np.sin(i*angle))
        models.append( (c, small_radius) )
    nmodels = len(models)
    npermodel = npoints//nmodels
    ground_truth = list()
    all_points   = list()
    for model in models:
        model_points = sim_sphere_cloud(npermodel, model, scatter, rng)
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


