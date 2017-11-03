from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import copy
import time
import matplotlib.pyplot as plt
from numpy import linalg
import SURE_tools

from NL_tools import *
from nt_toolbox.general import *
from nt_toolbox.signal import *
from patches_shapes import *


def preprocess(sigma, color=True):
    # path = "nt_toolbox/data/joseph_train.jpg"
    # path = "nt_toolbox/data/flowers.png"
    path = "nt_toolbox/data/Marguerite_et_joseph.jpg"
    # path = "nt_toolbox/data/EgliseBoisDeboutNorvege2.jpg"
    n = 500
    c = [100, 200]
    # f0 = load_image("nt_toolbox/data/lena.bmp")
    # f0 = rescale(f0[c[0]-n//2:c[0]+n//2, c[1]-n//2:c[1]+n//2])
    if color:
        f0 = load_image(path, n, grayscale = 0, square=False)
        print(f0.shape)
    else:
        f0 = load_image(path, n, grayscale = 1)
    y = add_noise(sigma, f0)
    return f0, y


class NL_Means(object):
    def __init__(self, sigma=0.02, color=True, w=3, kernel = "exponential", pca_dim = 25, locality_constraint = 5, tau = 0.05):
        self.sigma = sigma

        self.color = color
        f0, y = preprocess(self.sigma, color=self.color)
        self.f0 = f0
        self.y = y
        self.n, self.p = f0.shape[:2]

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
            self.patches = np.zeros((self.n, self.p, 3, self.w1, self.w1))
            for col in range(3):
                self.patches[:, :, col, :, :] = extract_all_patches(self.w, pict[:, :, col])


    def compute_PCA_patches(self):
        if self.color:
            self.H = self.y
            return
        flat_patches = patches_to_2D(self.patches, self.n, self.w1, self.p)
        flat_patches = flat_patches - np.tile(np.mean(flat_patches,0), (flat_patches.shape[0], 1))

        C = np.dot(flat_patches, np.transpose(flat_patches))
        [D,V] = linalg.eig(C)
        self.D = np.sort(D)[::-1]   # Diagonal matrix containing the eigen values of C
        I = np.argsort(D)[::-1]
        self.V = V[I,:]             # Ortogonal matrix of diagonalization of C

        Q = np.dot(np.transpose(self.V[: ,:self.pca_dim]), flat_patches)
        self.H = np.reshape(np.transpose(Q),(self.n, self.p, self.pca_dim),order="F")   #Descriptor of patches (n*n*pca_dim)



    def NLval_0(self, K, sel):
        if self.color:
            f_temp = self.y[sel[0],: , :]
            f_temp = f_temp[:, sel[1], :]
            KK = np.transpose(np.array([K, K, K]), [1,2,0])
            # Same weight for all color channels
            return np.sum(np.sum(KK * f_temp, axis=0), axis=0)
        else:
            f_temp = self.y[sel[0], :]
            return np.sum(K * f_temp[:, sel[1]])



    def NL_pixelwise_fourier(self, i, all_distances):
        distance = all_distances[i[0], i[1], :, :]
        if self.kernel == "exponential":
            return self.NLval_0(exponential_kernel(distance, self.tau), selection(i, self.q, self.n, self.p))
        else:
            print("%s is not an available kernel" %self.kernel)



    def apply_NL_Means_Fourier_0(self, all_distances):
        n, p = self.H.shape[:2]
        if self.color:
            res = np.zeros([n, p, 3])
            for x in range(n):
                for y in range(p):
                    res[x, y, :] = self.NL_pixelwise_fourier([x, y], all_distances)
        else:
            res = np.zeros([n,p])
            for x in range(n):
                for y in range(p):
                    res[x, y] = self.NL_pixelwise_fourier([x, y], all_distances)
        return res


    def compute_risks_maps(self):
        self.risk_maps = {}
        for theta in self.patches.keys():
            t0 = time.time()
            risk_map = SURE_tools.risk_map(self.patches[theta], self.all_distances[theta], self.tau, self.y, self.all_pictures[theta], self.kernel, self.q, self.sigma)

            self.risk_maps[theta] = risk_map
            t1 = time.time()
            print("risk map computed for theta = %.2f in %.1f seconds" %(theta, t1-t0))


    def compute_optimized_picture(self):
        optimized_picture = np.zeros_like(self.f0)
        for x in range(self.f0.shape[0]):
            for y in range(self.f0.shape[0]):
                theta_opt = 0
                risk_opt = 1000
                for theta in self.risk_maps.keys():
                    risk_xy = self.risk_maps[theta][x, y]
                    if risk_xy < risk_opt:
                        risk_opt = risk_xy
                        theta_opt = theta
                optimized_picture[x, y] = self.all_pictures[theta][x, y]
        return optimized_picture



    def apply_NL_Means_Fourier(self):
        self.patches = {}
        self.fourier_patches = {}
        self.all_distances = {}

        for theta in np.arange(0, 1, 0.33):
            theta = theta * 3.14159
            patch, fourier_patch = np.ma.conjugate(compute_rectangular_patch(self.n, self.w, theta, self.p))
            self.patches[theta] = patch
            self.fourier_patches[theta] = fourier_patch
            self.all_distances[theta] =  np.zeros((self.H.shape[0], self.H.shape[1], 2*self.q + 1, 2*self.q + 1))
        for dx in np.arange(-self.q, self.q+1):
            for dy in np.arange(-self.q, self.q+1):
                delta = [dx, dy]
                print(delta)
                distance_test = self.compute_distances(delta)
                distance_test_shifted = np.fft.ifftshift(distance_test)
                fourier_transform = np.fft.fft2(distance_test_shifted)

                for theta in self.patches.keys():
                    fourier_total = self.fourier_patches[theta] * fourier_transform
                    inverse_fourier = np.fft.ifftshift(np.real(np.fft.ifft2(fourier_total)))

                    self.all_distances[theta][:, :, dx + self.q, dy + self.q] = inverse_fourier
        self.all_pictures = {}
        imageplot(self.y, 'Origin image')
        plt.show()
        for i, theta in enumerate(self.patches.keys()):
            self.f_bar = self.apply_NL_Means_Fourier_0(self.all_distances[theta])
            self.all_pictures[theta] = self.f_bar
            imageplot(self.all_pictures[theta] , "corrected image, theta = %.2f Pi \n SNR = %.1f" %(theta/3.14159, snr(self.all_pictures[theta], self.f0)))
            plt.show()
            if i ==0:
                mean_picture = self.f_bar / float(len(self.patches.keys()))
            else:
                mean_picture += self.f_bar / float(len(self.patches.keys()))



        n_pict = len(self.all_pictures.keys())
        if self.color:
            # plt.figure(figsize = (5,5))
            for i, theta in enumerate(self.all_pictures.keys()):
                if i>3:
                    break
                imageplot(self.all_pictures[theta] , "corrected image, theta = %.2f Pi \n SNR = %.1f" %(theta/3.14159, snr(self.all_pictures[theta], self.f0)), [2, 2, i+1])
            plt.show()
            imageplot(mean_picture , "corrected mean image,\n SNR = %.1f" %(snr(mean_picture, self.f0)))
            plt.show()

            # print("Computing risk maps")
            # self.compute_risks_maps()
            # self.optimized_picture = self.compute_optimized_picture()
            # print("risk maps computed")
            # imageplot(self.optimized_picture , "corrected optimized image,\n SNR = %.1f" %(snr(self.optimized_picture, self.f0)))
            # plt.show()

        else:
            plt.figure(figsize = (5,5))
            imageplot(self.y, 'y', [1, 2, 1])
            imageplot(self.f_bar , "SNR : %.2f" %snr(self.f_bar, self.f0), [1, 2, 2])
            plt.show()




    def apply_NL_Means(self):
        [Y, X] = np.meshgrid(np.arange(0, self.p), np.arange(0, self.n))
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


    def compute_distances(self, delta):
        n, p = self.H.shape[:2]
        distance_delta = np.zeros((n, p))
        H_shifted = np.roll(self.H, -delta[0], axis = 0)
        H_shifted = np.roll(H_shifted, -delta[1], axis = 1)
        distance_delta = np.sum((self.H - H_shifted)**2, axis = 2)/(self.w1**2)
        return distance_delta


if __name__ == "__main__":
    color = True
    nl = NL_Means(tau=0.02, pca_dim=35, w=5, color=color, sigma=0.0, locality_constraint=5)
    # plot_picture(nl.y)
    t0 = time.time()
    if not color:
        nl.create_patches(nl.y)
    nl.compute_PCA_patches()
    nl.apply_NL_Means_Fourier()
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
