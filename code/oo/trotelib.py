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


def fit_bounding_box(points):
    if not len(points):
        return None
    if not len(points[0]):
        return None
    n = len(points[0])
    _min = tuple(np.min(tuple(p[i] for p in points)) for i in range(n))
    _max = tuple(np.max(tuple(p[i] for p in points)) for i in range(n))
    return tuple(zip(_min,_max))


def bounding_box_diameter(bounding_box):
    return np.linalg.norm( [ b[1] - b[0] for b in bounding_box] )


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

class TroteModel:
    def __init__(self, ambient_dim, nmpoints):
        """
        the two general parameters of all models are the dimension in which they live
        and the number of points needed to define the model.
        """
        self.ambient_dim = ambient_dim
        self.num_model_points = nmpoints
        self.model_dim = nmpoints - 1
        self.points = []

    def distance(self, points):
        """
        :return: the distance of the given points to this model
        """
        return []

    def ntests(self, npoints):
        """
        return the "number of tests" associated to constructing this model
        when using NFA. In general this will be just npoints^k, where k is
        the number of points needed to define the model.
        """
        ntests = special.binom(npoints, self.num_model_points)

    def draw_points(self,npoints,scatter):
        """
        draw npoints random points uniformly distributed in the region of points which are
        at most scatter distance from the model.
        :param npoints: now many points to draw
        :param scatter: how far the points can be
        :return: a list of points
        """
        return []

    def fit(self, list_of_points):
        """
        fit model to this list of points
        """
        return self

    def copy(self):
        _copy = TroteModel(self.ambient_dim,self.num_model_points)
        _copy.ambient_dim = self.ambient_dim
        _copy.num_model_points = self.num_model_points
        _copy.model_dim = self.model_dim
        # deep copy
        _copy.points = self.points[:]
        return _copy

    def translate(self, offset):
        """
        translate model
        """
        return self

    def rotate(self):
        """
        rotate model about first axis (not any rotation)
        """
        return self

    def find_aligned(self, points, max_dist):
        N = len(points)
        distances = self.distance(points)
        return list([points[i] for i in range(N) if distances[i] < max_dist])

    def ransac(self, list_of_points, num_samples, rng):
        """
        draw num_samples samples of this kind of model
        by performing nsamples draws of num_model_points from list_of_points
        with reposition and fitting a model to each drawn set
        :param list_of_points: input data points
        :param num_samples: number of samples to draw
        :param rng: random number generator
        :return: a list of fitted models
        """
        N = len(list_of_points)
        models = [self.copy() for i in range(num_samples)]
        idx = range(N)
        for i in range(num_samples):
            draw_idx = rng.choice(idx, size=self.num_model_points, replace=False)
            draw = [list_of_points[r] for r in draw_idx]
            models[i].fit(draw)
        return models



class AffineModel(TroteModel):

    def __init__(self, ambient_dim, affine_dim):
        super().__init__(ambient_dim,affine_dim+1)
        self.V = []
        self.W = []
        self.offset = []
        self.points = []

    def fit(self,list_of_points):
        k = len(list_of_points)
        if k < self.num_model_points:
            print(f"insufficient points {k} to build model requiring {self.num_model_points} points")
            return

        n = len(list_of_points[0])
        if n != self.ambient_dim:
            print(f"wrong dimension {n} when creating affine model of ambient dim {self.ambient_dim}")
            return

        self.points = list_of_points
        self.offset = list_of_points[0]
        m = self.model_dim
        if m > 0:
            # construct an orthogonal basis for the span of V and its orthogonal complement
            V = np.array(list_of_points[1:]) - self.offset
            Q = gram_schmidt(V)
            self.V = Q[:m, :]
            self.W = Q[m:, :]
        else:
            self.V = np.zeros((0, 0))  # a 0-dimensional affine subspace (the point x_0)
            self.W = np.eye(n)
        return self

    def distance(self,list_of_points):
        N = len(list_of_points)  # works with matrices and lists alike
        if N == 0:
            return []
        Xa = np.array(list_of_points) - self.offset
        Xp = Xa @ self.W.T
        return la.norm(Xp, axis=1)

    def project(self,list_of_points):
        N = len(list_of_points)  # works with matrices and lists alike
        if N == 0:
            return []
        Xa = np.array(list_of_points) - self.offset
        Xp = Xa @ self.V.T
        return Xp + self.offset

    def translate(self, offset):
        self.offset = [self.offset[i] + offset[i] for i in range(self.ambient_dim)]
        return self

    def rotate(self, angle):
        """
        rotate about first axis, not any arbitrary rotation
        """
        if angle != 0:
            R = np.eye(self.ambient_dim)
            R[0, 0] = R[1, 1] = np.cos(angle)
            R[0, 1] = np.sin(angle)
            R[1, 0] = -np.sin(angle)
            if self.model_dim > 0:
                self.V = self.V @ R
            self.W = self.W @ R

    def copy(self):
        _copy = AffineModel(self.ambient_dim,self.model_dim)
        _copy.ambient_dim = self.ambient_dim
        _copy.model_dim = self.model_dim
        _copy.num_model_points = self.num_model_points
        # deep copy
        _copy.points = self.points[:]
        _copy.V = self.V
        _copy.W = self.W
        _copy.offset = self.offset
        return _copy


class SphereModel(TroteModel):

    def __init__(self,ambient_dim):
        super().__init__(ambient_dim, ambient_dim + 1)
        self.center = []
        self.radius = []
        self.points = []

    def fit(self, list_of_points):
        pmat   = np.array(list_of_points)
        self.center = np.mean(pmat, axis=0)
        self.radius = np.linalg.norm(self.center - np.array(list_of_points[0]))
        self.points = list_of_points
        return self

    def distance(self,list_of_points):
        N = len(list_of_points)  # works with matrices and lists alike
        if N == 0:
            return []
        return np.abs(np.linalg.norm(np.array(list_of_points) - self.center, axis=1) - self.radius)

    def translate(self, offset):
        self.center = [self.center[i] + offset[i] for i in range(self.ambient_dim)]
        return self

    def rotate(self,angle):
        """
        does nothing, of course
        """
        return self


class PatchModel(AffineModel):
    def __init__(self,ambient_dim, model_dim):
        super().__init__(ambient_dim, model_dim)

    def distance(self,list_of_points):
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

        n = self.ambient_dim
        m = self.model_dim # can be at most n, may be less
        if m > 2: # pyramids and up, we don't do for now
            print("not implemented")
        if m == 0: # super easy, a point
            Xa = np.array(list_of_points) - self.offset
            return la.norm(Xa,axis=1)
        else:
            # a triangle or a segment
            Xa = np.array(list_of_points) - self.ofset
            if n-m > 0:
                Xortho = Xa @ self.W.T
                do =  la.norm(Xortho, axis=1)
            else:
                do = np.zeros(N)
            Xpara  = Xa @ self.V.T
            dp = np.zeros(N)
            # now the fine part: distance to the affine set within the affine space
            c = self.offset
            if m == 1: # easy, a segment
                # here we take the first point (c) as a reference
                # and a as the other point that defines the segment
                # we then express p as c + x * (a-c): x = (p-c)/(a-c)
                # if 0 <= x <= 1, the point p lies between c and a, so the distance is 0
                # if x < 0 or x > 1, the distance is corr. |x| or x-1
                #
                a = np.array(self.points[1])
                ac = c - a
                dac = la.norm(ac)
                # note that for the m=1 case, V is (a-c) normalized so that
                # the coefficient x is precisely Xpara*|c-a|
                dp = np.maximum(-Xpara,np.maximum(0,Xpara-dac))
                return np.sqrt(dp.ravel() ** 2 + do ** 2)
            if m == 2: # triangle in 3D, not so easy
                # we project onto the 2D plane where the points live
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
                a = np.array(self.points[1])
                b = np.array(self.points[2])
                # check interior
                AB = np.array([a-c,b-c]) # a and b as rows
                ABcoef = np.linalg.solve(AB.T,Xa.T)
                di = [ 1e20*(np.any(c < 0)+(np.sum(c) > 1)) for c in ABcoef.T]
                ab = PatchModel(2,2).fit([a,b])
                bc = PatchModel(2,2).fit([b,c])
                ca = PatchModel(2,2).fit([c,a])
                dab = ab.distance(list_of_points)
                dbc = bc.distance(list_of_points)
                dca = ca.distance(list_of_points)
                da  = [la.norm(p-a) for p in list_of_points]
                db  = [la.norm(p-a) for p in list_of_points]
                dc  = [la.norm(p-a) for p in list_of_points]
                dp  = np.minimum(np.minimum(di,np.minimum(dab,dbc)),np.minimum(np.minimum(dca,da),np.minimum(db,dc)))
                return np.sqrt(dp.ravel() ** 2 + do ** 2)


def fit_affine_model(list_of_points):
    k = len(list_of_points)
    if not k:
        return None
    n = len(list_of_points[0])
    if not n:
        return None
    m = k - 1
    return AffineModel(n,m).fit(list_of_points)

def build_affine_model(offset,V,W):
    k,n = V.shape
    model = AffineModel(n,k-1)
    model.offset = offset
    model.V = V
    model.W = W
    model.points = []
    return model


def fit_patch_model(list_of_points):
    k = len(list_of_points)
    if not k:
        return None
    n = len(list_of_points[0])
    if not n:
        return None
    m = k - 1
    return PatchModel(n,m).fit(list_of_points)


def build_patch_model(list_of_points):
    return fit_patch_model(list_of_points)


def fit_sphere_model(list_of_points):
    k = len(list_of_points)
    if not k:
        return None
    n = len(list_of_points[0])
    if not n:
        return None
    if k < n+1:
        return None
    return SphereModel(n).fit(list_of_points)


def build_sphere_model(center, radius):
    model = SphereModel(len(center))
    model.center = center
    model.radius = radius
    return model


def nfa_ks(points, model:TroteModel, scale, ntests=None, return_counts=False):
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
        ntests = special.binom(len(points), model.num_model_points)

    ambient_dim = len(points[0])  # infer ambient dimension from first data point
    res_dim = model.ambient_dim - model.model_dim  # infer orthogonal space dimension
    distances = list(d / scale for d in model.distance(points) if d <= scale)
    nclose = len(distances)
    if nclose <= model.num_model_points + 1:  # hay problemas con KStest con muy pocos puntos!
        if return_counts:
            return 10, nclose
        else:
            return 10
    _, pvalue = stats.kstest(distances, stats.powerlaw(res_dim).cdf, alternative='greater')
    if return_counts:
        return ntests * pvalue, nclose
    else:
        return ntests * pvalue


def ransac_nfa_uniscale_greedy(model_prototype:TroteModel, points, scale, nsamp, rng):
    """
    perform RANSAC of models akin to model_prototype
    and keep the ones with NFA below 1
    """
    N = len(points)
    detected_models = list()
    if not N:
        return detected_models

    candidates = model_prototype.ransac(points,nsamp,rng)
    cand_points = tuple(points)

    while len(cand_points):
        nfas = [nfa_ks(cand_points, cand, scale) for cand in candidates]
        best_idx = np.argmin(nfas)
        best_nfa  = nfas[best_idx]
        if best_nfa >= 1:
            break
        best_cand = candidates[best_idx]
        best_points = best_cand.find_aligned(cand_points,scale)
        detected_models.append((best_cand,best_points,best_nfa))
        #
        # remove aligned points
        #
        cand_points = subtract_points(cand_points,best_points)
    return detected_models


def ransac_nfa_multiscale_greedy(model_prototype:TroteModel, points, scale, factor, nsamp, rng, depth=0):
    detected_models = ransac_nfa_uniscale_greedy(model_prototype, points,scale,nsamp, rng)
    nmodels = len(detected_models)
    print(' '*depth,'depth',depth,'scale',scale,'points',len(points),'detected',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,p,s in detected_models:
            children = ransac_nfa_multiscale_greedy(model_prototype, points,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,points,children) )
        return model_nodes


def ransac_nfa_uniscale_rafa(model_prototype:TroteModel, points, scale, nsamp, rng):
    N = len(points)
    m = 1
    candidates = model_prototype.ransac(points,nsamp,rng)
    cand_models = list(candidates)
    detected_models = list()
    rem_points = list() # these are the excluding ALL significant models found so far
    #
    # The baseline (global ) NFAs are computed _once_
    #
    nfas = [nfa_ks(points, cand, scale, ntests=N ** 3) for cand in candidates]
    #
    # the candidates are sorted _once_ using their NFAs
    # the most significant have lower NFA, so the ascending order is ok
    #
    idx = np.argsort(nfas)
    nfas = [nfas[i] for i in idx]
    cand_models = [cand_models[i] for i in idx]
    cand_points = [find_aligned_points(c, points, scale) for c in cand_models]
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
            rem_nfa = nfa_ks(other_rem, other_model, scale)
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


def ransac_nfa_multiscale_rafa(model_prototype:TroteModel, points, scale, factor, nsamp, rng, depth=0):
    detected_models = ransac_nfa_uniscale_rafa(model_prototype, points,scale,nsamp, rng)
    nmodels = len(detected_models)
    print(' '*depth,'depth',depth,'scale',scale,'points',len(points),'detected',nmodels)
    if nmodels == 0:
        return ()
    else:
        model_nodes = list()
        for m,p,s in detected_models:
            children = ransac_nfa_multiscale_rafa(model_prototype, points,scale*factor,factor,nsamp, rng, depth=depth+1)
            model_nodes.append( (scale*factor,m,s,points,children) )
        return model_nodes

