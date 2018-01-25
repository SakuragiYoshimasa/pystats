#coding: utf-8
from numba import jit
import numpy as np

@jit('f8(f8[:], f8)')
def sum_of_square(data, mean):
    sos = 0
    for datum in data:
        sos += np.square(datum - mean)
    return sos

@jit('f8(f8[:], f8, f8[:])')
def sum_of_square_with_weight(data, mean, weight):
    sos = 0
    length = len(data)
    for i in range(length):
        sos += np.square(data[i] - mean) * weight[i]
    return sos
