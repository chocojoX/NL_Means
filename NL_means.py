from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import time
import matplotlib.pyplot as plt
from numpy import linalg

from NL_tools import *
from nt_toolbox.general import *
from nt_toolbox.signal import *


def preprocess(sigma, color=True):
    # path = "nt_toolbox/data/joseph_train.jpg"
    path = "nt_toolbox/data/hibiscus.bmp"
    n = 200
    c = [100, 200]
    # f0 = load_image("nt_toolbox/data/lena.bmp")
    # f0 = rescale(f0[c[0]-n//2:c[0]+n//2, c[1]-n//2:c[1]+n//2])
    if color:
        f0 = load_image(path, n, grayscale = 0)
    else:
        f0 = load_image(path, n, grayscale = 1)
    y = add_noise(sigma, f0)
    return f0, y


class NL_Means(object):
    def __init__(self, sigma=0.04, color=True, w=3, kernel = "exponential", pca_dim = 25, locality_constraint = 14, tau = 0.05):
        self.sigma = sigma

        self.color = color
        f0, y = preprocess(self.sigma, color=self.color)
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
        if not self.color:
            patches = extract_all_patches(self.w, pict)
            self.patches = patches
        else:
            self.patches = np.zeros((self.n, self.n, 3, self.w1, self.w1))
            for col in range(3):
                self.patches[:, :, col, :, :] = extract_all_patches(self.w, pict[:, :, col])


    def compute_PCA_patches(self):
        flat_patches = patches_to_2D(self.patches, self.n, self.w1)
        flat_patches = flat_patches - np.tile(np.mean(flat_patches,0), (flat_patches.shape[0], 1))

        C = np.dot(flat_patches, np.transpose(flat_patches))
        [D,V] = linalg.eig(C)
        self.D = np.sort(D)[::-1]   # Diagonal matrix containing the eigen values of C
        I = np.argsort(D)[::-1]
        self.V = V[I,:]             # Ortogonal matrix of diagonalization of C

        Q = np.dot(np.transpose(self.V[: ,:self.pca_dim]), flat_patches)
        self.H = np.reshape(np.transpose(Q),(self.n, self.n, self.pca_dim),order="F")   #Descriptor of patches (n*n*pca_dim)


    def compute_patch_distance(self, i):
        return np.sum((self.H - np.tile(self.H[i[0], i[1], :], (self.n, self.n, 1)))**2, 2)/(self.w1 * self.w1)


    def compute_kernel_globally(self, i, verbose=0):
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
        if self.color:
            f_temp = self.y[sel[0],: , :]
            KK = np.transpose(np.array([K, K, K]), [1,2,0])
            # Same weight for all color channels
            return np.sum(np.sum(KK * f_temp[:, sel[1], :], axis=0), axis=0)
        else:
            f_temp = self.y[sel[0],:]
            return np.sum(K * f_temp[:, sel[1]])


    def NL_pixelwise(self, i):

        distance = self.compute_local_patch_distance(i)
        if self.kernel == "exponential":
            return self.NLval_0(exponential_kernel(distance, self.tau), selection(i, self.q, self.n))
        else:
            print("%s is not an available kernel" %self.kernel)


    def apply_NL_Means_0(self, X, Y):
        n = len(X)
        p = len(Y)
        if self.color:
            res = np.zeros([n, p, 3])
            for x in range(n):
                for y in range(p):
                    res[x, y, :] = self.NL_pixelwise([x, y])
        else:
            res = np.zeros([n,p])
            for x in range(n):
                for y in range(p):
                    res[x, y] = self.NL_pixelwise([x, y])
        return res

    def apply_NL_Means(self):
        [Y, X] = np.meshgrid(np.arange(0, self.n), np.arange(0, self.n))
        self.f_bar = self.apply_NL_Means_0(X, Y)

        if self.color:
            plt.figure(figsize = (5,5))
            imageplot(self.y, 'Origin image', [1, 2, 1])
            imageplot(self.f_bar , "corrected image", [1, 2, 2])
            plt.show()
        else:
            plt.figure(figsize = (5,5))
            imageplot(self.y, 'y', [1, 2, 1])
            imageplot(self.f_bar , "SNR : %.2f" %snr(self.f_bar, self.f0), [1, 2, 2])
            plt.show()



if __name__ == "__main__":
    nl = NL_Means(tau=0.09, pca_dim=25, color=False)
    # plot_picture(nl.y)
    t0 = time.time()
    nl.create_patches(nl.y)
    nl.compute_PCA_patches()
    nl.apply_NL_Means()
    t1 = time.time()
    print("Total time to compute NL_means : %is" %(t1-t0))


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
