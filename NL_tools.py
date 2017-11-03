from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import matplotlib.pyplot as plt

from nt_toolbox.general import *
from nt_toolbox.signal import *


def plot_picture(f):
    plt.figure(figsize = (5,5))
    imageplot(f)
    plt.show()


def add_noise(sigma, f0, verbose = 0):
    h = f0.shape[0]
    w = f0.shape[1]
    if len(f0.shape) == 2:
        #Grayscale picture
        f = f0 + sigma*np.random.standard_normal((h,w))
    else:
        d = f0.shape[2]
        f = np.minimum(1, np.abs(f0 + sigma * np.random.standard_normal((h, w, d))))

        if verbose>2:
            plt.figure(figsize = (5,5))
            imageplot(f , "flute", [1, 2, 1])
            imageplot(f0 , "flute", [1, 2, 2])

        plt.show()

    return f


def extract_all_patches(w, pict):
    n = pict.shape[0]
    [X,Y,dX,dY]=np.meshgrid(np.arange(1,n+1),np.arange(1,n+1),np.arange(-w,w+1),np.arange(-w,w+1))
    X = X + dX
    Y = Y + dY

    X[X < 1] = 2-X[X < 1]
    Y[Y < 1] = 2-Y[Y < 1]
    X[X > n] = 2*n-X[X > n]
    Y[Y > n] = 2*n-Y[Y > n]

    I = (X-1) + (Y-1)*n
    for i in range(n//w):
        for j in range(n//w):
            I[i,j] = np.transpose(I[i,j])

    patch = np.ravel(pict)[I]
    return patch


def patches_to_2D(patches, n, w1):
    if len(patches.shape)==4:
        #Grayscale image
        return np.transpose((np.reshape(patches, (n*n,w1*w1), order="F")))
    else:
        # Color image
        one_chanel_patches = []
        for i in range(3):
            channel_patch = np.transpose((np.reshape(patches[:, :, i, :, :], (n*n,w1*w1), order="F")))
            one_chanel_patches.append(channel_patch)
        patches_2D = np.concatenate(one_chanel_patches)
        return patches_2D


def normalize(K):
    return K/np.sum(K)


def exponential_kernel(distance, tau, normalized=True):
    if normalized:
        return normalize(np.exp(-distance/(2*tau**2)))
    else:
        return np.exp(-distance/(2*tau**2))


def selection(i, q, n):
    return np.array((clamp(np.arange(i[0]-q,i[0] + q + 1), 0, n-1), clamp(np.arange(i[1]-q,i[1] + q + 1), 0, n-1)))


def get_coordinates_in_picture(x, y, n1, n2):
    x = np.abs(x); y = np.abs(y)
    if x>n1-1:
        x = x - (x-n1+1)
    if y>n2-1:
        y = y - (y-n2+1)
    return x, y
