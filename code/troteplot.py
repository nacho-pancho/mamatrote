#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as mplot3d
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

def plot_set_2d(ax,affine_set,color1,color2,show_ortho=False,length=1):

    c,V,W = affine_set
    m,n = V.shape

    ax.scatter([c[0]], [c[1]], color=color1, s = 1)

    for i in range(m):
        ax.plot((c[0], c[0] + length*V[i, 0]), (c[1], c[1] + length*V[i, 1]), color=color1,lw=0.5)
    if show_ortho:
        for j in range(n-m):
            ax.plot((c[0], c[0] + length*W[j,0]), (c[1], c[1] + length*W[j, 1]), color=color2,lw=0.5)

def plot_set_3d(ax,affine_set,color1,color2,show_ortho=False,length=1):

    c,V,W = affine_set
    m,n = V.shape

    ax.scatter([c[0]], [c[1]], [c[2]], color=color1)

    for i in range(m):
        ax.plot((c[0], c[0] + length*V[i, 0]), (c[1], c[1] + length*V[i, 1]), (c[2], c[2] + length*V[i, 2]), color=color1)
    if show_ortho:
        for j in range(n-m):
            ax.plot((c[0], c[0] + length*W[j,0]), (c[1], c[1] + length*W[j, 1]), (c[2], c[2] + length*W[j, 2]), color=color2)

def plot_set(ax,affine_set,color1,color2,show_ortho=False,length=1):
    """
    up to dimension 3
    """
    c,V,W = affine_set
    m,n = V.shape
    if n == 2:
        plot_set_2d(ax,affine_set,color1,color2,show_ortho,length)
    elif n == 3:
        plot_set_3d(ax,affine_set,color1,color2,show_ortho,length)


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
    #detmap = ListedColormap(colors=["red", "yellow", "cyan", "blue"])
    #detmap = LinearSegmentedColormap.from_list("pepe",colors=["red", "yellow", "cyan", "blue"])
    detmap = LinearSegmentedColormap.from_list("pepe",colors=[(0,"black"), (1,"blue")])
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.flipud(nfa), cmap='gray', extent=[x[0], x[-1], y[0], y[-1]], aspect='auto')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.colorbar()
    fname = title.lower().replace(' ','_').replace('=','_')
    plt.savefig(f'{fname}_img.png')
    plt.close(fig)


def plot_two_sets(affine_set_1, affine_set_2, points_1, points_2, ran):
    fig = plt.figure(figsize=(12,12))
    n = points_1.shape[1]
    if n == 2:
        ax = fig.add_subplot()
        plot_set(ax, affine_set_1, color1='red', color2='red')
        plot_set(ax, affine_set_2, color1='blue', color2='blue')
        ax.scatter(points_1[:,0],points_1[:,1],color='orange')
        ax.scatter(points_2[:,0],points_2[:,1],color='cyan')
        #ax.xlim(-ran,ran)
        #ax.ylim(-ran,ran)
    elif n == 3:
        ax = fig.add_subplot(projection='3d')
        plot_set(ax, affine_set_1, color1='red', color2='red')
        plot_set(ax, affine_set_2, color1='blue', color2='blue')
        ax.scatter(points_1[:,0],points_1[:,1],points_1[:,2], color='orange')
        ax.scatter(points_2[:,0],points_2[:,1],points_1[:,2], color='cyan')
        ax.view_init(elev=70,azim=120)
        #ax.xlim(-ran,ran)
        #ax.ylim(-ran,ran)
        #ax.zlim(-ran,ran)
