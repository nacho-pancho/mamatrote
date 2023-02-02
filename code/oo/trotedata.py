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


def sim_points(npoints, bounding_box, _rng):
    """
    simulate npoints points uniformly distributed within the specified bounding box
    """
    m = len(bounding_box) # bounding box is a list of pairs x[i]_min, x[i]_max where i is the dim
    return tuple(
        tuple(
            _rng.uniform(low=bounding_box[i][0],high=bounding_box[i][1])
            for i in range(m)
        )
        for j in range(npoints)
    )

def inside_bounding_box(points, bounding_box):
    n = len(points)
    if n == 0:
        return []
    m = len(points[0])
    if m == 0:
        return []
    y = np.zeros(n)
    for i,p in enumerate(points):
        inside = True
        for j in range(m):
            if p[j] < bounding_box[j][0] or p[j] > bounding_box[j][1]:
                inside = False
                break
        y[i] = inside
    return y

def bounding_box_diameter(bounding_box):
    return np.linalg.norm( [ b[1] - b[0] for b in bounding_box] )

def sim_background_points(npoints, bounding_box, rng):
    """
    simulate npoints points uniformly distributed within the specified bounding box
    """
    return sim_points(npoints,bounding_box,rng)

def sim_affine_model(affine_dim, bounding_box, rng):
    """
    Simulate an affine set in arbitrary dimension and with arbitrary subdimension
    :param bonding_box: sampled points uniformly sampled within this bounding box
    :param affine_dim: dimensions of the affine subset
    :return: a pair x_0,(v_1,v_2,...) where x_0 is the offset  and v_1... are the vectors that define the direction of the set
    """
    points  = sim_points(affine_dim+1,bounding_box,rng)
    ambient_dim = len(bounding_box)
    return AffineModel(ambient_dim,affine_dim).fit(points)

def sim_patch_model(bounding_box, patch_dim, rng):
    points  = sim_points(patch_dim+1,bounding_box,rng)
    ambient_dim = len(bounding_box)
    return PatchModel(ambient_dim,patch_dim).fit(points)


def sim_affine_points(_affine_set, _num_points, bounding_box, scatter, _rng, scatter_distro=None):
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
    c,V,W,P = _affine_set.offset, _affine_set.V, _affine_set.W, _affine_set.points
    n = len(c)
    m = len(V)
    if scatter_distro is None:
        scatter_distro = build_scatter_distribution(n - m,_rng)

    bbox_diameter = bounding_box_diameter(bounding_box)
    n = len(c)
    m = len(V)
    list_of_points = list()
    for i in range(_num_points):
        a = _rng.normal(size=(n-m)) # anisotropic
        norm = np.linalg.norm(a)
        d = scatter*scatter_distro(1)
        a *= d/norm
        if len(V) > 0:
            # sample a point along the affine model so that it
            # is uniformly distributed within the bounding box
            b = _rng.uniform(low=-bbox_diameter, high=bbox_diameter, size = m)
            aux = c + b @ V
            while not inside_bounding_box([aux],bounding_box)[0]:
                b = _rng.uniform(low=-bbox_diameter, high=bbox_diameter, size = m)
                aux = c + b @ V
            x = aux + a @ W
        else:
            x =  c + a @ W
        list_of_points.append(x)
    return list_of_points



def sim_patch_points(_num_points, _patch, _scatter, _rng):
    """
    simulate a cloud of points whose distribution is uniform within the region defined
    by all points whose distance to the patch is less than a scatter value.
    We sample values uniformly from a bounding box and keep those inside
    """
    c,V,W,P = _patch.offset, _patch.V, _patch.W, _patch.P
    bbox0  = fit_bounding_box(P)
    bbox = [(r[0]-_scatter,r[1]+_scatter) for r in bbox0 ]
    n = len(c)
    m = len(V)
    _sim_points = list()
    _rem_points = _num_points
    while _rem_points > 0:
        _raw_points = sim_background_points(_rem_points*2,bbox,_rng)
        _distances  = _patch.distance(_raw_points)
        _inner_points = [p for p,d in zip (_raw_points,_distances) if d <= _scatter]
        _sim_points.extend(_inner_points)
        _rem_points -= len(_inner_points)
    if len(_sim_points) > _num_points:
        _sim_points = _sim_points[:_num_points]
    return _sim_points


def sim_sphere_model(bounding_box, rng):
    """
    Simulate a sphere model
    :param ambient_dim: space where the affine set lives
    :return: a pair x_0,(v_1,v_2,...) where x_0 is the offset  and v_1... are the vectors that define the direction of the set
    """
    ambient_dim = len(bounding_box) # bounding box is a list of pairs x[i]_min, x[i]_max where i is the dim
    model =  SphereModel(ambient_dim)
    model.center = tuple(
            rng.uniform(low=bounding_box[i][0],high=bounding_box[i][1])
            for i in range(ambient_dim)
        )
    # sample radius so that it is within the bounding box
    max_rad = np.min(
        np.array(
            [
                min(np.abs(model.center[i] - bounding_box[i][0]), np.abs(model.center[i] - bounding_box[i][1]))
                for i in range(ambient_dim)
            ]
        )
    )
    model.radius = rng.uniform(low=0, high=max_rad)


def sim_sphere_points(_num_points, _sphere, _scatter, _rng):
    """
    simulate a cloud of points whose distribution is uniform within the region defined
    by all points whose distance to the patch is less than a scatter value.
    We sample values uniformly from a bounding box and keep those inside
    """
    c,r = _sphere.center, _sphere.radius
    c = np.array(c)
    n = len(c)
    baux1 = c-r-_scatter
    baux2 = c+r+_scatter
    bbox = [(baux1[i],baux2[i]) for i in range(n) ]
    _sim_points = list()
    _rem_points = _num_points
    while _rem_points > 0:
        _raw_points = sim_background_points(_rem_points*2,bbox,_rng)
        _distances  = _sphere.distance(_raw_points)
        _inner_points = [p for p,d in zip (_raw_points,_distances) if d <= _scatter]
        _sim_points.extend(_inner_points)
        _rem_points -= len(_inner_points)
    if len(_sim_points) > _num_points:
        _sim_points = _sim_points[:_num_points]
    return _sim_points

#====================================================================================
# TOY DATASETS
#====================================================================================
#

def azucarlito(npoints,scatter,rng):
    # kind of Anarchy symbol with double horizontal bar
    models = list()
    lp = [[6.5,9],[3,14]]
    models.append(AffineModel(2,1).fit(lp))
    lp = [[5.5,6.5],[1,3]]
    models.append(AffineModel(2,1).fit(lp))
    lp = [[4.5,9],[5,14]]
    models.append(AffineModel(2,1).fit(lp))
    lp = [[3.5,9],[4,14]]
    models.append(AffineModel(2,1).fit(lp))
    k = len(models)
    ground_truth = list()
    npermodel = npoints // (k+1)
    all_points = list()
    bounding_box = [(0,15),(0,15)]
    for model in models:
        model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
        ground_truth.append( (model,model_points) )
        all_points.extend(model_points)
    return all_points,ground_truth


def waffle(npoints, scatter, rng, size = 10, nlines=4):
    models = list()
    for i in range(nlines):
        a = (i*size/(nlines-1),0)
        b = (i*size/(nlines-1),size)
        row = AffineModel(2, 1).fit([a,b])
        models.append(row)
        a = (0,i*size/(nlines-1))
        b = (size,i*size/(nlines-1))
        col = AffineModel(2, 1).fit([a,b])
        models.append( col )
    nmodels = len(models)
    npermodel = npoints//nmodels
    ground_truth = list()
    all_points   = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=0, high=10)
        model_points = sim_affine_points(model, npermodel, rng, scatter=scatter, model_distro=model_distro)
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
        models.append( AffineModel(2,1).fit([a,b]) )
        if not i:
            break
    nmodels = len(models)
    npermodel = npoints//nmodels
    ground_truth = list()
    all_points   = list()
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=0, high=10)
        model_points = sim_affine_points(model, npermodel, rng, scatter=scatter, model_distro=model_distro)
        ground_truth.append((model, model_points))
        all_points.extend(model_points)
    return all_points, ground_truth


def some_rings(npoints,scatter,rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    npermodel = npoints // 5
    # ojo
    model = SphereModel(2)
    model.center = (2,6)
    model.radius = 1.5
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # pupila
    model = SphereModel(2)
    model.center = (2,6)
    model.radius = 0.5
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # otro ojo
    model = SphereModel(2)
    model.center = (6,6)
    model.radius = 2
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth

def carucha(npoints,rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    npermodel = npoints // 5
    # cara
    model = SphereModel(2)
    model.center = (5,5)
    model.radius = 5
    model_points = sim_sphere_points(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # nariz
    model = SphereModel(2)
    model.center = (5,4)
    model.radius = 1
    model_points = sim_sphere_points(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # ojo
    model = SphereModel(2)
    model.center = (3,6)
    model.radius = 2.2
    model_points = sim_sphere_points(npermodel, model, 0.1, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # pupila
    model = SphereModel(2)
    model.center = (3.2,6)
    model.radius = 0.6
    model_points = sim_sphere_points(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # otro ojo
    model = SphereModel(2)
    model.center = (7,6)
    model.radius = 2
    model_points = sim_sphere_points(npermodel, model, 0.1, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)
    # pupila
    model = SphereModel(2)
    model.center = (7,6)
    model.radius = 0.7
    model_points = sim_sphere_points(npermodel, model, 0.2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth


def collar(npoints, scatter, rng, big_radius = 5, small_radius = 1, rings = 7):
    models = list()
    angle = 2.0*np.pi/rings
    for i in range(rings):
        c = (big_radius * np.cos(i*angle), big_radius * np.sin(i*angle))
        model = SphereModel(2)
        model.center = c
        model.radius = small_radius
        models.append( model )
    nmodels = len(models)
    npermodel = npoints//nmodels
    ground_truth = list()
    all_points   = list()
    for model in models:
        model_points = sim_sphere_points(npermodel, model, scatter, rng)
        ground_truth.append((model, model_points))
        all_points.extend(model_points)
    return all_points, ground_truth


def clusters_and_lines(npoints, scatter, rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    bounding_box = ((0,10),(0,10))
    nmodels = 4
    npermodel = npoints // nmodels
    # clusters
    c = [(2,5)]
    model = AffineModel(2,0).fit(c)
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    c = [(8,5)]
    model = fit_affine_model(c)
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    # lines
    a = (0,0)
    b = (10,10)
    model = fit_affine_model([a, b])
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    a = (0,10)
    b = (10,0)
    model = fit_affine_model([a, b])
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth


def clusters_and_circles(npoints, scatter, rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    bounding_box = ((0,10),(0,10))
    nmodels = 4
    npermodel = npoints // nmodels

    # clusters
    c = [(2,5)]
    model = fit_affine_model(c)
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter*2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    c = [(8,5)]
    model = fit_affine_model(c)
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter*2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    # circles
    model = [(5,2),1.5]
    models.append(model)
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    model = [(5,7),2]
    models.append(model)
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth


def lines_and_circles(npoints, scatter, rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    bounding_box = ((0,10),(0,10))
    nmodels = 4
    npermodel = npoints // nmodels

    # lines
    a = (0,0)
    b = (10,10)
    model = fit_affine_model([a, b])
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    a = (0,10)
    b = (10,0)
    model = fit_affine_model([a, b])
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    # circles
    model = [(5,2),1.5]
    models.append(model)
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    model = [(5,8),1]
    models.append(model)
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    return all_points, ground_truth


def clusters_and_lines_and_circles(npoints, scatter, rng):
    models = list()
    ground_truth = list()
    all_points   = list()
    bounding_box = ((0,10),(0,10))
    nmodels = 4
    npermodel = npoints // nmodels

    # clusters
    c = [(2,5)]
    model = fit_affine_model(c)
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter*2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    c = [(8,5)]
    model = fit_affine_model(c)
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter*2, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    # lines
    a = (0,0)
    b = (10,10)
    model = fit_affine_model([a, b])
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    a = (0,10)
    b = (10,0)
    model = fit_affine_model([a, b])
    models.append(model)
    model_points = sim_affine_points(model, npermodel, bounding_box, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    # circles
    model = SphereModel(2)
    model.center = (5,2)
    model.radius = 1.5
    models.append(model)
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
    ground_truth.append((model, model_points))
    all_points.extend(model_points)

    model = SphereModel(2)
    model.center = (5,8)
    model.radius = 1
    models.append(model)
    model_points = sim_sphere_points(npermodel, model, scatter, rng)
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


