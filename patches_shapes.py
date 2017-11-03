import numpy as np


def compute_square_patch(n, w):
    x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
    [Y, X] = np.meshgrid(x, x)
    h = np.maximum(np.abs(X), np.abs(Y))
    h[h<=w] = 1
    h[h>w] = 0

    hF = np.real(np.fft.fft2(h))
    return h, hF


def compute_gaussian_patch(n, w):
    x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
    [Y, X] = np.meshgrid(x, x)
    hh = np.exp(-X**2-Y**2)
    h = np.maximum(np.abs(X), np.abs(Y))
    h[h<=w] = 1 * hh[h<=w]
    h[h>w] = 0

    hF = np.real(np.fft.fft2(h))
    return h, hF


def compute_rectangular_patch(n, w, theta, p=None):
    if p is None:
        p=n
    x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
    y = np.concatenate( (np.arange(0,p/2), np.arange(-p/2,0)) );
    [Y, X] = np.meshgrid(y, x)
    c = np.cos(theta)
    s = np.sin(theta)
    XX = c*X+s*Y
    YY = -s*X + c*Y
    h = np.maximum(np.abs(1.5*XX), np.abs(YY/1.5))
    h[h<=w] = 1
    h[h>w] = 0

    hF = np.real(np.fft.fft2(h))
    return h, hF


def compute_horizontal_patch(n, w, p=None):
    if p is None:
        p=n
    x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
    y = np.concatenate( (np.arange(0,p/2), np.arange(-p/2,0)) );
    [Y, X] = np.meshgrid(y, x)
    h = np.maximum(np.abs(1.5*X), np.abs(Y/1.5))
    h[h<=w] = 1
    h[h>w] = 0

    hF = np.real(np.fft.fft2(h))
    return h, hF

def compute_vertical_patch(n, wp=None):
    if p is None:
        p=n
    x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
    y = np.concatenate( (np.arange(0,p/2), np.arange(-p/2,0)) );
    [Y, X] = np.meshgrid(y, x)
    h = np.maximum(np.abs(X/2), np.abs(Y*2))
    h[h<=w] = 1
    h[h>w] = 0

    hF = np.real(np.fft.fft2(h))
    return h, hF
