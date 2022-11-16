#!/usr/bin/env python3
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


def sim_affine_cloud(affine_set, num_points, distro1, distro2):
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
    x_0,V,W = affine_set
    n = len(x_0)
    m = len(V)
    b = distro1((num_points,m))
    c = distro2((num_points,n-m))
    if len(V) > 0:
        return x_0 + b @ V + c @ W
    else:
        return x_0 + c @ W


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
    d2 = lambda x: rng.laplace(size=x,scale=0.01)
    test_points = sim_affine_cloud(affine_set, N, d1,d2)
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
