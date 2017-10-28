from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import matplotlib.pyplot as plt

from NL_tools import *
from nt_toolbox.general import *
from nt_toolbox.signal import *


def preprocess():
    n = 128
    c = [100,200]
    f0 = load_image("nt_toolbox/data/lena.bmp")

    f0 = rescale(f0[c[0]-n//2:c[0]+n//2, c[1]-n//2:c[1]+n//2])
    y = add_noise(0.04, f0)

    return f0, y



if __name__ == "__main__":
    f0, y = preprocess()
    plot_picture(y)
