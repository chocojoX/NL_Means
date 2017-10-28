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
