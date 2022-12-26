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
