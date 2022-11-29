#!/usr/bin/env python3
from scipy import stats

"""
TROTELIB
========

Main library for the acronym-free universal detection algorithm
The routines in this module can be divided into three parts:
   * Sample points from a affine set models (e.g., point, line, plane, etc.).
     The points are sampled within a given distance to the true model according to some distribution
   * Estimate models (affine sets) from points
   * Compute distance to affine sets
   * Compute detection scores using various methods
"""
import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import stats
from scipy import special

def build_scatter_distribution(r, theta=0):
    """
    Construct a probability distribution over the ball of radius in ambient
    space R^r so that the distribution of the distance to the origin is uniform.
    This is a power law distribution with scale parameter s and power r-1
    :param r: dimension of the ball
    :param s: radius of the ball
    :param theta: if > 0, adds an  e^{-qx factor} to the density, which introduces a decay.
                since the support is [0,1], we need to correct for the truncated domain
    :return:
    """
    if theta <= 0:
        return lambda size: np.random.power(r,size=size)
    else:
        return lambda size: np.random.power(r-theta,size=size)
        #return lambda size: stats.gengamma.rvs(a=r,c=theta,size=size)


def build_affine_set(list_of_points):
    """

    :param X: list of points that define the affine set; the first point (x_0) is the offset, the rest, x_1,x_2,
              define the span of the set
    :return: a triplet (x_0,V,W) where x_0 is the offset, V is a matrix whose rows are (x_1-x_0,x_2-x_0,...),
             and W is a basis for the orthogonal complement of V
    """
    #
    # convert X to numpy array, if it is a list
    #
    _rng = random.default_rng(123098)
    x_0 = np.array(list_of_points[0])
    n = len(x_0)
    m = len(list_of_points)-1
    if m > 0:
        # construct an orthogonal basis for the span of V and its orthogonal complement
        V = np.array(list_of_points[1:]) - x_0
        _A = _rng.normal(size=(n,n))
        _A[:m,:] = V
        _Q,_ = la.qr(_A.T) # QR operates on columns; we have rows; thus the transpostion
        _Q = _Q.T
        V = _Q[:m,:] # basis for the linear part of the affine set
        W = _Q[m:,:] # orthogonal complement
    else:
        V = np.zeros((0,0)) # a 0-dimensional affine subspace (the point x_0)
        W,_ = la.qr(_rng.normal(size=(n,n)))
        W = W.T
    return x_0, V, W

def build_affine_set_relative_to(affine_set,dist=0,angle=0):
    """
    Given an affine set, build another one so that it is at a given angle
    respect to the first dimension (in the first dimension only, not generic rotation)
    or either parallel at a given distance (along the first direction of W)
    """
    c,V,W = affine_set
    m,n = V.shape
    n = len(c)
    #print("m=",m)
    if angle != 0:
        R      = np.eye(n)
        #print(V.shape,W.shape, R.shape)
        R[0,0] = R[1,1] = np.cos(angle)
        R[0,1] = np.sin(angle)
        R[1,0] = -np.sin(angle)
        if m > 0:
            V2 = V @ R
        else:
            V2 = V
        W2 = W @ R
    else:
        V2 = V
        W2 = W
    
    if dist != 0:
        c2 = c + dist * W[0,:]
    else:
        c2 = c
    return (c2,V2,W2)


def distance_to_affine(list_of_points, affine_set, P=None):
    """
    Compute the distance from a set of points to the subspace
    spanned by a list of vectors given by the rows of a matrix V.

    :param X: list of points whose distance is to be measured
    :param V: list of vectors defining the subspace
    :param P:  projection operator onto the subspace, if available
    :return: Euclidean distance from each point in x to the subspace:  ||x-Px||_2
    """
    N = len(list_of_points) # works with matrices and lists alike
    if N == 0:
        return []
    x_0, V, W = affine_set
    Xa = np.array(list_of_points) - x_0
    #print(Xa.shape,W.shape)
    Xp = Xa @ W.T
    return la.norm(Xp,axis=1)


def sim_affine_set(ambient_dim,affine_dim,distro):
    """
    Simulate an affine set in arbitrary dimension and with arbitrary subdimension
    :param ambient_dim: space where the affine set lives
    :param affine_dim: dimensions of the affine subset
    :return: a pair x_0,(v_1,v_2,...) where x_0 is the offset  and v_1... are the vectors that define the direction of the set
    """
    x0 = distro(ambient_dim)
    V = distro((affine_dim+1,ambient_dim))
    return build_affine_set(V)


def sim_affine_cloud(_affine_set, _num_points, scatter = 1.0, model_distro=None, scatter_distro=None):
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
    c,V,W = _affine_set
    m,n = V.shape
    rng = random.default_rng()
    if model_distro is None:
        model_distro = lambda x: rng.uniform(size=x, low=-scatter*10, high=scatter*10)
    if scatter_distro is None:
        scatter_distro = build_scatter_distribution(n - m)

    n = len(c)
    m = len(V)
    b = model_distro((_num_points,m))
    a0 = np.random.normal(size=(_num_points,n-m)) # anisotropic
    _norms = np.linalg.norm(a0,axis=1)
    _deltas = scatter*scatter_distro(_num_points)
    a = np.outer(_deltas/_norms, np.ones(n-m)) * a0
    if len(V) > 0:
        x =  c + b @ V + a @ W
        return x
    else:
        x =  c + a @ W
        return x

def sim_background_points(npoints,n,bg_scale=1, bg_dist=None,seed=42):
    rng = random.default_rng(seed=42)
    if bg_dist is None:
        bg_dist = lambda x: rng.uniform(size=x)
    return bg_dist((npoints, n))

def ransac_affine(points, m, k, seed=42):
    """
    Create k candidate models from random samples of m+1 n-dimensional points
    :param points: input data points
    :param m: dimension of affine space
    :param k: number of samples to draw
    :return: a list of candidate affine models
    """
    N,n = points.shape
    rng = random.default_rng(seed=seed)
    models = list()
    idx = range(N)
    for i in range(k):
        chosen_ones = rng.choice(idx,size=m+1,replace=False)
        list_of_points = [points[r,:] for r in chosen_ones ]
        sampled_model = build_affine_set(list_of_points)
        models.append(sampled_model)
    return models

def find_aligned_points(points, affine_set, distance, scale):
    distances = distance(points,affine_set)
    N,n = points.shape
    return list([points[i] for i in range(N) if distances[i] < scale])

def nfa_ks(data, model, model_dim, model_nparam, distance, scale):
    """
    Compute the Kolmogorov-Smirnoff-based NFA score for a set of points w.r.t. a model (for a given scale).
    
    :param data: data points
    :param model: a candidate model
    :param model_dim: dimension of the model
    :param model_nparam: number of parameters required to define the model
    :param distance: distance function from the data points to a model
    :param scale: scale of the analysis; here it is the width of the bands
    :return: the NFA detection score for the model given the data points and the analysis scale
    """
    ambient_dim = len(data[0])        # infer ambient dimension from first data point
    res_dim = ambient_dim - model_dim # infer orthogonal space dimension 
    distances = list(d/scale for d in distance(data, model) if d <= scale)
    NT = special.binom(len(data), model_nparam)
    if len(distances) <= 0:
        return NT
    _, pvalue = stats.kstest(distances, stats.powerlaw(res_dim).cdf, alternative='greater')
    return NT * pvalue


#==========================================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = random.default_rng(42)
    m = 1
    n = 2
    N = 1000
    distro0 = lambda x: rng.uniform(size=x, low=0, high=1)
    affine_set  = sim_affine_set(n,m,distro0)
    d1 = lambda x: rng.uniform(size=x,low=-2,high=2)
    d2 = build_scatter_distribution(n - m)
    scatter = 0.02
    test_points = sim_affine_cloud(affine_set, N, d1, d2, scatter)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(test_points[:,0],test_points[:,1],color='gray',alpha=0.05)
    x_0, V, W = affine_set
    a = x_0
    b = x_0 + V[0]
    c = x_0 + W[0]
    plt.scatter([a[0]],[a[1]],color='black')
    plt.plot([a[0],b[0]],[a[1],b[1]],color='red')
    plt.plot([a[0],c[0]],[a[1],c[1]],color='green')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    d = distance_to_affine(test_points,affine_set)
    plt.subplot(1,2,2)
    plt.hist(d)
    plt.show()

    scale = 0.1
    affine_set2 = sim_affine_set(n, m, distro0)
    points = np.copy(test_points[-2:])
    points += rng.normal(0, 0.2*scale, size=points.shape)
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
    plt.figure()
    plt.scatter(test_points[:,0], test_points[:,1], color='gray',alpha=0.2)
    plt.plot([a1[0],b1[0]],[a1[1],b1[1]],color='red',label=f"NFA={nfa1}")
    plt.plot([a2[0],b2[0]],[a2[1],b2[1]],color='green',label=f"NFA={nfa2}")
    plt.plot([a3[0],b3[0]],[a3[1],b3[1]],color='blue',label=f"NFA={nfa3}")
    plt.legend()
    plt.show()
