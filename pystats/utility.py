#coding: utf-8
from numba import jit
import numpy as np

@jit
def sum_of_square(data, mean):
    sos = 0
    for datum in data:
        sos += np.square(datum - mean)
    return sos
