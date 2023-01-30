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

def build_scatter_distribution(r, rng, theta=0):
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
        return lambda size: rng.power(r,size=size)
    else:
        return lambda size: rng.power(r-theta,size=size)

def bounding_box(points):
    if not len(points):
        return None
    if not len(points[0]):
        return None
    n = len(points[0])
    _min = tuple(np.min(tuple(p[i] for p in points)) for i in range(n))
    _max = tuple(np.max(tuple(p[i] for p in points)) for i in range(n))
    return tuple(zip(_min,_max))

def gram_schmidt(V):
    m,n = V.shape
    A = np.empty((n,n))
    A[:m, :] = V[:m,:]/np.outer(np.linalg.norm(V[:m,:],axis=1),np.ones(n))
    A[m:, :] = np.random.random((n-m,n))
    G = A[:m, :].T @ A[:m, :]
    for i in range(m,n):
        A[i, :] -= G @ A[i, :]
        A[i,:]  /= np.linalg.norm(A[i,:])
        G += np.outer(A[i, :],A[i, :])
    return A


def find_aligned_points(points, affine_set, distance, scale):
    distances = distance(points,affine_set)
    N = len(points)
    return list([points[i] for i in range(N) if distances[i] < scale])


def subtract_points(a,b):
    """
    utility to subract using sets of points
    """
    a_aux  = [tuple(c) for c in a]
    b_aux  = [tuple(c) for c in b]
    return list(set(a_aux).difference(set(b_aux)))

"""
=============================
AFFINE SETS
=============================
"""
def build_affine_set(list_of_points):
    """

    :param X: list of points that define the affine set; the first point (x_0) is the offset, the rest, x_1,x_2,
              define the span of the set
    :return: a tuple (x_0,V,W,p) where x_0 is the offset, V is a matrix whose rows are (x_1-x_0,x_2-x_0,...),
             W is a basis for the orthogonal complement of V, and p is the list of poits used to create the set
             which may later be used to simulate
    """
    #
    # convert X to numpy array, if it is a list
    #
    # it is ok that this one is fixed; it's just auxiliary
    _rng = np.random.default_rng(42)
    x_0 = list_of_points[0]
    n = len(x_0)
    m = len(list_of_points) - 1
    if m > 0:
        # construct an orthogonal basis for the span of V and its orthogonal complement
        V = np.array(list_of_points[1:]) - x_0
        Q = gram_schmidt(V)
        V = Q[:m, :]
        W = Q[m:, :]
        # _A = _rng.normal(size=(n,n))
        # _A[:m,:] = V
        # _Q,_ = la.qr(_A.T) # QR operates on columns; we have rows; thus the transpostion
        # _Q = _Q.T
        # V = _Q[:m,:] # basis for the linear part of the affine set
        # W = _Q[m:,:] # orthogonal complement
    else:
        V = np.zeros((0, 0))  # a 0-dimensional affine subspace (the point x_0)
        W, _ = la.qr(_rng.normal(size=(n, n)))
        W = W.T
    return x_0, V, W, list_of_points

def build_patch(list_of_points):
    """
    The patch models have exactly the same data as the affine models,
    but they are confined to the convex hull of the vertices, which we
    generically call a "patch", as in a mesh.
    The first point in the list is kept as an offset.
    :param list_of_points:
    :return: a patch model
    """
    return build_affine_set(list_of_points)

def build_shpere(list_of_points):
    pmat = np.array(list_of_points)
    c    = np.mean(pmat,axis=0)
    r = np.linalg.norm(c-np.array(list_of_points[0]))
    return (c,r)

def project_onto_affine(list_of_points, affine_set):
    N = len(list_of_points)  # works with matrices and lists alike
    if N == 0:
        return []
    x_0, V, W, P = affine_set
    Xa = np.array(list_of_points) - x_0
    Xp = Xa @ V.T
    return Xp + x_0

def distance_to_patch(list_of_points, patch):
    """
    The distance from p to a patch is sqrt(a^2+b^2) where
    a is the distance from p  to the affine set containing the patch and
    b is the distance from the projection of p onto the patch.
    The second is an easy task if the patch is 1D (a segment) or 2D (a triangle),
    otherwise we need to resort to a linear program.
    :param list_of_points: self explanatory
    :param patch: the patch
    :return: the distance of p to the patch
    """
    N = len(list_of_points)  # works with matrices and lists alike
    if N == 0:
        return []

    c, V, W, P = patch
    m, n = V.shape # dim of patch and ambient space
    if m > 2:
        print("not implemented")
        exit(1)
    elif m == 0:
        if m == 0: # super easy, a point
            Xa = np.array(list_of_points) - c
            return la.norm(Xa,axis=1)
    else:
        # a triangle or a segment
        Xa = np.array(list_of_points) - c
        if n-m > 0:
            Xortho = Xa @ W.T
            do =  la.norm(Xortho, axis=1)
        else:
            do = np.zeros(N)
        Xpara  = Xa @ V.T
        dp = np.zeros(N)
        # now the fine part: distance to the affine set within the affine space
        if m == 1: # easy, a segment
            # here we take the first point (c) as a reference
            # and a as the other point that defines the segment
            # we then express p as c + x * (a-c): x = (p-c)/(a-c)
            # if 0 <= x <= 1, the point p lies between c and a, so the distance is 0
            # if x < 0 or x > 1, the distance is corr. |x| or x-1
            #
            a = np.array(P[1])
            ac = c - a
            dac = la.norm(ac)
            # note that for the m=1 case, V is (a-c) normalized so that
            # the coefficient x is precisely Xpara*|c-a|
            #for i in range(N):
            #    print(i,c,Xa[i,:],list_of_points[i],Xpara[i])
            dp = np.maximum(-Xpara,np.maximum(0,Xpara-dac))
            d =  np.sqrt(dp.ravel()**2 + do**2)
            return d
            #return np.sqrt(dp ** 2)
        if m == 2: # triangle, not so easy
            #
            # given the 3 vertices a, b, c, and a point there are 6 distances:
            # the point to the three segments a-b, b-c, c-a
            # the point to the three vertices
            # we can compute all six using the case m== 1 and m == 0 (point)
            # if the point is inside the triangle, the distance is 0
            # we can check this by computing the unique representation of p-c in terms of (a-c,b-c)
            # and checking whether it is a convex combination (coefficients >= 0 and <= 1)
            # if it is outside, it  is the smallest of all six
            # notice that there is some redundancy here as the distances to the segments
            # depend on the vertices too, but it's easier this way
            #
            a = np.array(P[1])
            b = np.array(P[2])
            # check interior
            AB = np.array([a-c,b-c]) # a and b as rows
            ABcoef = np.linalg.solve(AB.T,Xa.T)
            di = [ 1e20*(np.any(c < 0)+(np.sum(c) > 1)) for c in ABcoef.T]
            ab = build_patch([a,b])
            bc = build_patch([b,c])
            ca = build_patch([c,a])
            dab = distance_to_patch(list_of_points,ab)
            dbc = distance_to_patch(list_of_points,bc)
            dca = distance_to_patch(list_of_points,ca)
            da  = [la.norm(p-a) for p in list_of_points]
            db  = [la.norm(p-a) for p in list_of_points]
            dc  = [la.norm(p-a) for p in list_of_points]
            return np.minimum(np.minimum(di,np.minimum(dab,dbc)),np.minimum(np.minimum(dca,da),np.minimum(db,dc)))

def distance_to_sphere(list_of_points, sphere):
    """
    This is not the distance to a disk, but the distance to the surface of the sphere.
    That means that the points inside the sphere will have a positive distance to it.
    :param list_of_points: self explanatory
    :param patch: the patch
    :return: the distance of p to the patch
    """
    N = len(list_of_points)  # works with matrices and lists alike
    c,r = sphere
    if N == 0:
        return []
    return np.abs(np.linalg.norm(np.array(list_of_points)-c,axis=1)-r)

def build_affine_set_relative_to(affine_set, dist=0, angle=0):
    """
    Given an affine set, build another one so that it is at a given angle
    respect to the first dimension (in the first dimension only, not generic rotation)
    or either parallel at a given distance (along the first direction of W)
    """
    c, V, W, P = affine_set
    m, n = V.shape
    n = len(c)
    if angle != 0:
        R = np.eye(n)
        R[0, 0] = R[1, 1] = np.cos(angle)
        R[0, 1] = np.sin(angle)
        R[1, 0] = -np.sin(angle)
        if m > 0:
            V2 = V @ R
        else:
            V2 = V
        W2 = W @ R
    else:
        V2 = V
        W2 = W

    if dist != 0:
        c2 = c + dist * W[0, :]
    else:
        c2 = c
    return (c2, V2, W2, V)


def distance_to_affine(list_of_points, affine_set, P=None):
    """
    Compute the distance from a set of points to the subspace
    spanned by a list of vectors given by the rows of a matrix V.

    :param X: list of points whose distance is to be measured
    :param V: list of vectors defining the subspace
    :param P:  projection operator onto the subspace, if available
    :return: Euclidean distance from each point in x to the subspace:  ||x-Px||_2
    """
    N = len(list_of_points)  # works with matrices and lists alike
    if N == 0:
        return []
    x_0, V, W, P = affine_set
    Xa = np.array(list_of_points) - x_0
    Xp = Xa @ W.T
    return la.norm(Xp, axis=1)


def ransac_affine(points, m, k, _rng):
    """
    Create k candidate models from random samples of m+1 n-dimensional points
    :param points: input data points
    :param m: dimension of affine space
    :param k: number of samples to draw
    :return: a list of candidate affine models
    """
    N = len(points)
    models = list()
    idx = range(N)
    for i in range(k):
        chosen_ones = _rng.choice(idx,size=m+1,replace=False)
        list_of_points = [points[r] for r in chosen_ones ]
        sampled_model = build_affine_set(list_of_points)
        models.append(sampled_model)
    return models


def ransac_sphere(points, m, k, _rng):
    """
    Create a sphere from n+1 points in ambient dimension n sampled at random
    :param points: input data points
    :param m: dimension of affine space
    :param k: number of samples to draw
    :return: a list of candidate affine models
    """
    N = len(points)
    models = list()
    idx = range(N)
    for i in range(k):
        chosen_ones = _rng.choice(idx,size=m+1,replace=False)
        list_of_points = [points[r] for r in chosen_ones ]
        sampled_model = build_shpere(list_of_points)
        models.append(sampled_model)
    return models

def ransac_patch(points, m, k, rng):
    return ransac_affine(points, m, k, rng)


def nfa_ks(points, model, model_dim, model_nparam, distance, scale, ntests=None, return_counts=False):
    """
    Compute the Kolmogorov-Smirnoff-based NFA score for a set of points w.r.t. a model (for a given scale).

    :param points: data points
    :param model: a candidate model
    :param model_dim: dimension of the model
    :param model_nparam: number of parameters required to define the model
    :param distance: distance function from the data points to a model
    :param scale: scale of the analysis; here it is the width of the bands
    :return: the NFA detection score for the model given the data points and the analysis scale
    """
    if ntests is None:
        ntests = special.binom(len(points), model_nparam)

    ambient_dim = len(points[0])  # infer ambient dimension from first data point
    res_dim = ambient_dim - model_dim  # infer orthogonal space dimension
    distances = list(d / scale for d in distance(points, model) if d <= scale)
    nclose = len(distances)
    if nclose <= model_nparam + 1:  # hay problemas con KStest con muy pocos puntos!
        if return_counts:
            return 10, nclose
        else:
            return 10
    _, pvalue = stats.kstest(distances, stats.powerlaw(res_dim).cdf, alternative='greater')
    if return_counts:
        return ntests * pvalue, nclose
    else:
        return ntests * pvalue


def ransac_nfa_affine_uniscale_greedy(points,scale,nsamp,rng):
    N = len(points)
    m = 1
    detected_models = list()
    if not N:
        return detected_models

    candidates = ransac_affine(points,m,nsamp,rng)
    cand_points = tuple(points)

    while len(cand_points):
        nfas = [nfa_ks(cand_points, cand, m, m + 1, distance_to_affine, scale, ntests=N**3) for cand in candidates]
        best_idx = np.argmin(nfas)
        best_nfa  = nfas[best_idx]
        if best_nfa >= 1:
            break
        best_cand = candidates[best_idx]
        best_points = find_aligned_points(cand_points,best_cand,distance_to_affine,scale)
        detected_models.append((best_cand,best_points,best_nfa))
        #
        # remove aligned points
        #
        cand_points = subtract_points(cand_points,best_points)
    return detected_models


def ransac_nfa_affine_multiscale_greedy(points, scale, factor, nsamp, rng, depth=0):
    detected_models = ransac_nfa_affine_uniscale_greedy(points,scale,nsamp, rng)
    nmodels = len(detected_models)
    print(' '*depth,'depth',depth,'scale',scale,'points',len(points),'detected',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,p,s in detected_models:
            children = ransac_nfa_affine_multiscale_greedy(points,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,points,children) )
        return model_nodes


def ransac_nfa_affine_uniscale_rafa(points,scale,nsamp,rng):
    N = len(points)
    m = 1
    candidates = ransac_affine(points,m,nsamp,rng)
    cand_models = list(candidates)
    detected_models = list()
    rem_points = list() # these are the excluding ALL significant models found so far
    #
    # The baseline (global ) NFAs are computed _once_
    #
    nfas = [nfa_ks(points, cand, m, m + 1, distance_to_affine, scale, ntests=N ** 3) for cand in candidates]
    #
    # the candidates are sorted _once_ using their NFAs
    # the most significant have lower NFA, so the ascending order is ok
    #
    idx = np.argsort(nfas)
    nfas = [nfas[i] for i in idx]
    cand_models = [cand_models[i] for i in idx]
    cand_points = [find_aligned_points(points, c, distance_to_affine, scale) for c in cand_models]
    cand_models = list(zip(cand_models,nfas,cand_points))
    # now we repeat
    #
    excluded_points = list()
    while len(cand_models):
        best_cand,best_nfa,best_points   = cand_models[0]
        #print("NFA of best model",best_nfa)
        if best_nfa >= 1:
            break
        detected_models.append((best_cand,best_points,best_nfa))
        excluded_points.extend(best_points)
        filtered_models = list()
        for t in range(1,len(cand_models)):
            other_model,other_nfa,other_points = cand_models[t]
            #
            # remove the best candidate points from this model
            #
            other_rem = subtract_points(other_points,excluded_points)
            #print(f"{t:5} other points {len(other_points):6} other non-redundant points {len(other_rem):6}",end=" ")
            if len(other_rem) <= m+2:
                #print(" not enough points")
                continue
            #
            # see if it is still significant
            #
            rem_nfa = nfa_ks(other_rem, other_model, m, m + 1, distance_to_affine, scale, ntests=N ** 2)
            #print(f"orig NFA {other_nfa:16.4f} NFA of non-redundant points {rem_nfa:16.4f}",end=" ")
            #
            # if it is, it is the new top
            #
            if rem_nfa < 1:
                #print("-> non-redundant")
                filtered_models.append((other_model,other_nfa,other_points))
            else:
                pass
                #print("-> redundant")
            # if other_nfa >= 1, the top is incremented but the other model is _not_ added to the list
        #
        #
        # we continue the analysis with the filtered models
        #print("redundant ",len(cand_models)-len(filtered_models),"non-redundant ",len(filtered_models))
        cand_models = filtered_models
    return detected_models


def ransac_nfa_affine_multiscale_rafa(points, scale, factor, nsamp, rng, depth=0):
    detected_models = ransac_nfa_affine_uniscale_rafa(points,scale,nsamp, rng)
    nmodels = len(detected_models)
    print(' '*depth,'depth',depth,'scale',scale,'points',len(points),'detected',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,p,s in detected_models:
            children = ransac_nfa_affine_multiscale_rafa(points,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,points,children) )
        return model_nodes



def ransac_nfa_sphere_uniscale_greedy(points,scale,nsamp,rng):
    N = len(points)
    m = 1
    detected_models = list()
    if not N:
        return detected_models

    candidates = ransac_sphere(points,m,nsamp,rng)
    cand_points = tuple(points)

    while len(cand_points):
        nfas = [nfa_ks(cand_points, cand, m, m + 1, distance_to_sphere, scale, ntests=N**3) for cand in candidates]
        best_idx = np.argmin(nfas)
        best_nfa  = nfas[best_idx]
        if best_nfa >= 1:
            break
        best_cand = candidates[best_idx]
        best_points = find_aligned_points(cand_points,best_cand,distance_to_sphere,scale)
        detected_models.append((best_cand,best_points,best_nfa))
        #
        # remove aligned points
        #
        cand_points = subtract_points(cand_points,best_points)
    return detected_models


def ransac_nfa_sphere_multiscale_greedy(points, scale, factor, nsamp, rng, depth=0):
    detected_models = ransac_nfa_sphere_uniscale_greedy(points,scale,nsamp, rng)
    nmodels = len(detected_models)
    print(' '*depth,'depth',depth,'scale',scale,'points',len(points),'detected',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,p,s in detected_models:
            children = ransac_nfa_sphere_multiscale_greedy(points,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,points,children) )
        return model_nodes


def ransac_nfa_sphere_uniscale_rafa(points,scale,nsamp,rng):
    N = len(points)
    m = 1
    candidates = ransac_sphere(points,m,nsamp,rng)
    cand_models = list(candidates)
    detected_models = list()
    rem_points = list() # these are the excluding ALL significant models found so far
    #
    # The baseline (global ) NFAs are computed _once_
    #
    nfas = [nfa_ks(points, cand, m, m + 1, distance_to_sphere, scale, ntests=N ** 3) for cand in candidates]
    #
    # the candidates are sorted _once_ using their NFAs
    # the most significant have lower NFA, so the ascending order is ok
    #
    idx = np.argsort(nfas)
    nfas = [nfas[i] for i in idx]
    cand_models = [cand_models[i] for i in idx]
    cand_points = [find_aligned_points(points, c, distance_to_sphere, scale) for c in cand_models]
    cand_models = list(zip(cand_models,nfas,cand_points))
    # now we repeat
    #
    excluded_points = list()
    while len(cand_models):
        best_cand,best_nfa,best_points   = cand_models[0]
        #print("NFA of best model",best_nfa)
        if best_nfa >= 1:
            break
        detected_models.append((best_cand,best_points,best_nfa))
        excluded_points.extend(best_points)
        filtered_models = list()
        for t in range(1,len(cand_models)):
            other_model,other_nfa,other_points = cand_models[t]
            #
            # remove the best candidate points from this model
            #
            other_rem = subtract_points(other_points,excluded_points)
            #print(f"{t:5} other points {len(other_points):6} other non-redundant points {len(other_rem):6}",end=" ")
            if len(other_rem) <= m+2:
                #print(" not enough points")
                continue
            #
            # see if it is still significant
            #
            rem_nfa = nfa_ks(other_rem, other_model, m, m + 1, distance_to_sphere, scale, ntests=N ** 2)
            #print(f"orig NFA {other_nfa:16.4f} NFA of non-redundant points {rem_nfa:16.4f}",end=" ")
            #
            # if it is, it is the new top
            #
            if rem_nfa < 1:
                #print("-> non-redundant")
                filtered_models.append((other_model,other_nfa,other_points))
            else:
                pass
                #print("-> redundant")
            # if other_nfa >= 1, the top is incremented but the other model is _not_ added to the list
        #
        #
        # we continue the analysis with the filtered models
        #print("redundant ",len(cand_models)-len(filtered_models),"non-redundant ",len(filtered_models))
        cand_models = filtered_models
    return detected_models


def ransac_nfa_sphere_multiscale_rafa(points, scale, factor, nsamp, rng, depth=0):
    detected_models = ransac_nfa_sphere_uniscale_rafa(points,scale,nsamp, rng)
    nmodels = len(detected_models)
    print(' '*depth,'depth',depth,'scale',scale,'points',len(points),'detected',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,p,s in detected_models:
            children = ransac_nfa_sphere_multiscale_rafa(points,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,points,children) )
        return model_nodes


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
    test_points = sim_affine_cloud(affine_set, N, rng, scatter, d1, d2)
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
