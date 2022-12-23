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
    bg_points = tuple([(30*p-5) for p in sim_background_points(npermodel,2, rng)])
    ground_truth.append( ("background",bg_points) )
    all_points.extend(bg_points)
    for model in models:
        model_distro = lambda x: rng.uniform(size=x, low=-10, high=10)
        model_points = sim_affine_cloud(model, npermodel, rng, scatter=scatter,model_distro=model_distro)
        ground_truth.append( (model,model_points) )
        all_points.extend(model_points)
    return all_points,ground_truth