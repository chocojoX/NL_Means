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
    def __init__(self, sigma=0.04, color=False, w=3, kernel = "exponential", pca_dim = 25, locality_constraint = 14, tau = 0.05):
        self.sigma = sigma

        self.color = False
        f0, y = preprocess(self.sigma)
        self.f0 = f0
        self.y = y
        self.n = f0.shape[0]

        """ nl-means params"""
        self.w = 3    # Half width of patches
        self.w1 = 2*self.w + 1
        self.pca_dim = pca_dim   # Dimension of the patch descriptor
        self.q = locality_constraint    # Locality constraint
        self.kernel = kernel
        self.tau = tau


    def create_patches(self, pict):
        patches = extract_all_patches(self.w, pict)
        self.patches = patches


    def compute_PCA_patches(self):
        flat_patches = patches_to_2D(self.patches, self.n, self.w1)
        flat_patches = flat_patches - np.tile(np.mean(flat_patches,0),(self.w1**2, 1))

        C = np.dot(flat_patches, np.transpose(flat_patches))
        [D,V] = linalg.eig(C)
        self.D = np.sort(D)[::-1]   # Diagonal matrix containing the eigen values of C
        I = np.argsort(D)[::-1]
        self.V = V[I,:]             # Ortogonal matrix of diagonalization of C

        Q = np.dot(np.transpose(self.V[: ,:self.pca_dim]), flat_patches)
        self.H = np.reshape(np.transpose(Q),(self.n, self.n, self.pca_dim),order="F")   #Descriptor of patches (n*n*pca_dim)

    def compute_patch_distance(self, i):
        return np.sum((self.H - np.tile(self.H[i[0], i[1], :], (self.n, self.n, 1)))**2, 2)/(self.w1 * self.w1)


    def compute_kernel_globally(self, verbose=0):
        i = [83,72]
        distance = self.compute_patch_distance(i)
        if self.kernel == "exponential":
            K = exponential_kernel(distance, self.tau)
        else:
            print("%s is not a valid kernel" %self.kernel)

        if verbose>2:
            plt.figure(figsize = (10,10))
            imageplot(distance, 'D', [1, 2, 1])
            imageplot(K, 'K', [1, 2, 2])
            plt.show()


    def compute_local_patch_distance(self, i):
        sel = selection(i, self.q, self.n)
        H1 = (self.H[sel[0],:,:])
        H2 = (H1[:,sel[1],:])
        return np.sum((H2 - np.tile(self.H[i[0],i[1],:], (len(sel[0]), len(sel[1]), 1)))**2,2)/self.w1*self.w1


    def NLval_0(self, K, sel):
        f_temp = self.y[sel[0],:]
        return np.sum(K*f_temp[:, sel[1]])


    def NL_pixelwise(self, i):

        distance = self.compute_local_patch_distance(i)
        if self.kernel == "exponential":
            return self.NLval_0(exponential_kernel(distance, self.tau), selection(i, self.q, self.n))
        else:
            print("%s is not an available kernel" %self.kernel)


    def apply_NL_Means_0(self, X, Y):
        n = len(X)
        p = len(Y)
        res = np.zeros([n,p])
        for x in range(n):
            for y in range(p):
                res[x, y] = self.NL_pixelwise([x, y])
        return res

    def apply_NL_Means(self):
        [Y, X] = np.meshgrid(np.arange(0, self.n), np.arange(0, self.n))
        self.f_bar = self.apply_NL_Means_0(X, Y)


        plt.figure(figsize = (5,5))
        imageplot(self.y, 'y', [1, 2, 1])
        imageplot(self.f_bar , "SNR : %.2f" %snr(self.f_bar, self.f0), [1, 2, 2])
        plt.show()



if __name__ == "__main__":
    nl = NL_Means(tau=0.15)
    # plot_picture(nl.y)
    nl.create_patches(nl.y)
    nl.compute_PCA_patches()
    nl.compute_kernel_globally()
    nl.apply_NL_Means()


    # plt.figure(figsize = (5,5))
    # P = nl.patches
    # n = 128
    # for i in range(16):
    #     x = np.random.randint(n)
    #     y = np.random.randint(n)
    #     imageplot(P[x, y], '', [4, 4, i+1])

    # plt.figure(figsize = (5,5))
    # for i in range(16):
    #     imageplot(abs(np.reshape(nl.V[:,i], (nl.w1, nl.w1))), '', [4, 4, i+1])
    # plt.show()
