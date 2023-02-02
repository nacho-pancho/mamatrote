#!/usr/bin/env python3
"""
pretty plotting datasets and stuff
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as mplot3d
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import patches
import numpy as np

from trotelib import *

def plot_affine_model_3d(ax, affine_set, color1, color2, show_ortho=False, length=1):

    c,V,W = affine_set
    m,n = V.shape

    ax.scatter([c[0]], [c[1]], [c[2]], color=color1)

    for i in range(m):
        ax.plot((c[0], c[0] + length*V[i, 0]), (c[1], c[1] + length*V[i, 1]), (c[2], c[2] + length*V[i, 2]), color=color1)
    if show_ortho:
        for j in range(n-m):
            ax.plot((c[0], c[0] + length*W[j,0]), (c[1], c[1] + length*W[j, 1]), (c[2], c[2] + length*W[j, 2]), color=color2)


def plot_affine_model_2d(ax, affine_set, width, bounding_box, color, border=False):
    """
    up to dimension 3
    """
    c,V,W,points = affine_set.offset, affine_set.V, affine_set.W, affine_set.points
    m,n = V.shape
    diameter = bounding_box_diameter(bounding_box)
    if m == 0: # ball
        ax.add_patch(patches.Circle(c,width))
    elif m == 1:
        if border:
            edgecolor = color
        else:
            edgecolor = (0,0,0,0)
        a = (c[0] - diameter*V[0, 0], c[1] - diameter*V[0, 1])
        b = (c[0] + diameter*V[0, 0], c[1] + diameter * V[0, 1])
        a1 = (a[0] - width * W[0,0], a[1] - width*W[0,1])
        a2 = (a[0] + width * W[0, 0], a[1] + width * W[0, 1])
        b1 = (b[0] - width * W[0, 0], b[1] - width * W[0, 1])
        b2 = (b[0] + width * W[0, 0], b[1] + width * W[0, 1])
        if border:
            plt.plot((a1[0],b1[0]),(a1[1],b1[1]),color=color)
            plt.plot((a2[0],b2[0]),(a2[1],b2[1]),color=color)
            plt.plot((a1[0],a2[0]),(a1[1],a2[1]),color=color)
            plt.plot((b1[0],b2[0]),(b1[1],b2[1]),color=color)
        x = (a1[0],a2[0],b2[0],b1[0])
        y = (a1[1],a2[1],b2[1],b1[1])
        ax.fill( x, y, color=color,edgecolor=edgecolor )


def plot_patch_model_2d(ax, patch, scatter, bounding_box, color, border=False):
    """
    not really working; using the affine one for now
    """
    c,V,W,P = patch.offset, patch.V, patch.W, patch.points
    length = np.linalg.norm(np.array(P[1])-np.array(P[0]))
    width = scatter
    m,n = V.shape

    if m == 0: # ball
        ax.add_patch(patches.Circle(c,width))
    elif m == 1:
        if border:
            edgecolor = color
        else:
            edgecolor = (0,0,0,0)
        a = (c[0] - length*V[0, 0], c[1] - length*V[0, 1])
        b = (c[0] + length*V[0, 0], c[1] + length * V[0, 1])
        a1 = (a[0] - width * W[0,0], a[1] - width*W[0,1])
        a2 = (a[0] + width * W[0, 0], a[1] + width * W[0, 1])
        b1 = (b[0] - width * W[0, 0], b[1] - width * W[0, 1])
        b2 = (b[0] + width * W[0, 0], b[1] + width * W[0, 1])
        if border:
            plt.plot((a1[0],b1[0]),(a1[1],b1[1]),color=color)
            plt.plot((a2[0],b2[0]),(a2[1],b2[1]),color=color)
            plt.plot((a1[0],a2[0]),(a1[1],a2[1]),color=color)
            plt.plot((b1[0],b2[0]),(b1[1],b2[1]),color=color)
        x = (a1[0],a2[0],b2[0],b1[0])
        y = (a1[1],a2[1],b2[1],b1[1])
        ax.fill( x, y, color=color,edgecolor=edgecolor )


def plot_sphere_model_2d(ax, sphere, scatter, bounding_box, color=(1, 0, 0, 0.2), border=False):
    """
    not really working; using the affine one for now
    """
    c,r = sphere.center, sphere.radius
    if border:
        edgecolor = color
    else:
        edgecolor = (0,0,0,0)

    wedge_radius = r + scatter
    wedge_width  = 2*scatter
    if r > scatter:
        wedge = patches.Wedge(c,wedge_radius,0,360,width=wedge_width,edgecolor=edgecolor,color=color)
    else:
        wedge = patches.Wedge(c,wedge_radius,0,360,edgecolor=edgecolor,color=color)
    ax.add_patch( wedge )


def plot_model(ax,model:TroteModel, scatter, bounding_box, color=(1, 0, 0, 0.2), border=False):
    if isinstance(model,AffineModel):
        plot_affine_model_2d(ax, model, scatter, bounding_box, color=color, border=border)
    elif isinstance(model,SphereModel):
        plot_sphere_model_2d(ax, model, scatter, bounding_box, color=color, border=border)
    elif isinstance(model,PatchModel):
        plot_patch_model_2d(ax, model, scatter, bounding_box, color=color, border=border)
    else:
        print("dont know how to plot this model")


def plot_scores_2d(x,xlabel,y,ylabel,nfa,title):
    from matplotlib.colors import LightSource

    X, Y = np.meshgrid(x,y)
    Z = nfa.T
    fig = plt.figure(figsize=(10,10))
    ax3 = fig.add_subplot(projection='3d')
    thresmap = ListedColormap(["red", "red", "blue", "blue"])

    ls = LightSource(250, 70,hsv_max_val=1) # azimuth, alt
    colors = thresmap(Z)
    Znorm = Z-np.min(Z)
    Znorm  /= np.max(Znorm)
    shaded_colors = ls.shade_rgb(colors,Znorm,vert_exag=0.02)
    ax3.plot_surface(X,Y,Z, edgecolor='black', lw=0.25,facecolors=shaded_colors,antialiased=False,shade=False)
    ax3.set(xlabel=xlabel,ylabel=ylabel,zlabel='-log(NFA)',title=title)
    ax3.view_init(elev=45,azim=120)
    fname = title.lower().replace(' ','_').replace('=','_')
    plt.savefig(f'{fname}_3d.svg')
    plt.close(fig)


def plot_scores_img(y,ylabel,x,xlabel,nfa,title):
    #detmap = LinearSegmentedColormap.from_list("pepe",colors=[(0,"black"), (1,"blue")])
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.flipud(nfa), cmap='gray', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fname = title.lower().replace(' ','_').replace('=','_')
    plt.savefig(f'{fname}_img.png')
    plt.close(fig)



def plot_uniscale_ransac(ax, all_points, models, scores, model_points, scale, bounding_box=None):
    """
    :return:
    """
    if bounding_box is None:
        bounding_box = fit_bounding_box(all_points)
    max_score = np.max(scores)
    cmap = cm.get_cmap("jet")
    for model,score,mpoints in zip(models,scores,model_points):
        color = cmap(score/max_score)
        color = (*color[:3],0.2)
        plot_model(ax, model, scale, bounding_box, color)
        mpoints = np.array(mpoints)
        plt.scatter(mpoints[:, 0], mpoints[:, 1], color="gray", s=4, alpha=0.5)
        plt.scatter(mpoints[0, 0], mpoints[0, 1], alpha=1,s=0.01) # hack para que el colorbar no quede transparente
    xmin = np.min([p[0] for p in all_points])
    xmax = np.max([p[0] for p in all_points])
    ymin = np.min([p[1] for p in all_points])
    ymax = np.max([p[1] for p in all_points])
    xlen = xmax-xmin
    ylen = ymax-ymin
    maxlen = max(xlen,ylen)
    plt.xlim(xmin,xmin+maxlen)
    plt.ylim(ymin,ymin+maxlen)
    plt.title('detected models')


def plot_multiscale_ransac(ax, model_node, plot_leaves, plot_single_parents, plot_branches):
    """
    :return:
    """
    cmap = cm.get_cmap("jet")
    _scale, _model, _score, _points, _children = model_node
    #if True:
    plot_it = False
    if len(_children) == 0 and plot_leaves:
        _color = (1,0,0,0.05)
        plot_it = True
    elif len(_children) == 1 and plot_single_parents:
        _color = (0,1,0,0.05)
        plot_it = True
    elif plot_branches:
        _color = (0,0,1,0.05)
        plot_it = True
    #print('plotting ',_scale,_score,len(_points),len(_children))
    if plot_it:
        plot_model(ax, _model, 50, _scale, _color)
        plot_points(ax,_points, color="gray", size=4, alpha=0.5)
        plot_points(ax,_points, alpha=1, size=0.01)  # hack para que el colorbar no quede transparente
    for node in _children:
        # TEMPORAL
        #
        # FIN TEMPORAL
        plot_multiscale_ransac(ax, node, plot_leaves, plot_single_parents, plot_branches)
    return ax


def plot_points(ax,list_of_points,size=4,alpha=1,color='black'):
    mat = np.array(list_of_points)
    plt.scatter(mat[:,0],mat[:,1],color=color,s=size,alpha=alpha)


def plot_ring(ax):
    n, radii = 50, [.7, .95]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]

    ax.fill(np.ravel(xs), np.ravel(ys), edgecolor='#348ABD')
