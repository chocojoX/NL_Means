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


def add_noise(sigma, f0):
    n = f0.shape[0]
    f = f0 + sigma*np.random.standard_normal((n,n))
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


def patches_to_2D(patches):
    return np.transpose((np.reshape(patches, (n*n,w1*w1), order="F")))
