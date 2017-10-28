from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import matplotlib.pyplot as plt
from numpy import linalg

from NL_tools import *
from nt_toolbox.general import *
from nt_toolbox.signal import *


def preprocess(sigma):
    n = 128
    c = [100,200]
    f0 = load_image("nt_toolbox/data/lena.bmp")

    f0 = rescale(f0[c[0]-n//2:c[0]+n//2, c[1]-n//2:c[1]+n//2])
    y = add_noise(sigma, f0)

    return f0, y


class NL_Means(object):
    def __init__(self, sigma=0.04, color=False, w=3):
        self.sigma = sigma

        self.color = False
        f0, y = preprocess(self.sigma)
        self.f0 = f0
        self.y = y
        self.n = f0.shape[0]

        """ nl-means params"""
        self.w = 3
        self.w1 = 2*self.w + 1
        self.pca_dim = 25


    def create_patches(self, pict):
        patches = extract_all_patches(self.w, pict)
        self.patches = patches


    def compute_PCA_patches(self):
        flat_patches = patches_to_2D(self.patches)
        flat_patches = flat_patches - np.tile(np.mean(flat_patches,0),(self.w1**2, 1))

        C = np.dot(flat_patches, np.transpose(flat_patches))
        [D,V] = linalg.eig(C)
        self.D = np.sort(D)[::-1]   # Diagonal matrix containing the eigen values of C
        I = np.argsort(D)[::-1]
        self.V = V[I,:]             # Ortogonal matrix of diagonalization of C

        Q = np.dot(np.transpose(self.V[: ,:self.pca_dim]), flat_patches)
        self.H = np.reshape(np.transpose(Q),(self.n, self.n, self.pca_dim),order="F")   #Descriptor of patches (n*n*pca_dim)




if __name__ == "__main__":
    nl = NL_Means()
    plot_picture(nl.y)

    plt.figure(figsize = (5,5))

    nl.create_patches(nl.y)
    P = nl.patches
    n = 128
    for i in range(16):
        x = np.random.randint(n)
        y = np.random.randint(n)
        imageplot(P[x, y], '', [4, 4, i+1])
    plt.show()
