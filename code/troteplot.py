#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as mplot3d


def plot_set_2d(ax,affine_set,color1,color2):

    c,V,W = affine_set
    m,n = V.shape

    ax.scatter([c[0]], [c[1]], color=color1)

    for i in range(m):
        ax.plot((c[0], c[0] + V[i, 0]), (c[1], c[1] + V[i, 1]), color=color1)
    for j in range(n-m):
        ax.plot((c[0], c[0] + W[j,0]), (c[1], c[1] + W[j, 1]), color=color2)

def plot_set_3d(ax,affine_set,color1,color2):

    c,V,W = affine_set
    m,n = V.shape

    ax.scatter([c[0]], [c[1]], [c[2]], color=color1)

    for i in range(m):
        ax.plot((c[0], c[0] + V[i, 0]), (c[1], c[1] + V[i, 1]), (c[2], c[2] + V[i, 2]), color=color1)
    for j in range(n-m):
        ax.plot((c[0], c[0] + W[j,0]), (c[1], c[1] + W[j, 1]), (c[2], c[2] + W[j, 2]), color=color2)

def plot_set(ax,affine_set,color1,color2):
    """
    up to dimension 3
    """
    c,V,W = affine_set
    m,n = V.shape
    if n == 2:
        plot_set_2d(ax,affine_set,color1,color2)
    elif n == 3:
        plot_set_3d(ax,affine_set,color1,color2)


