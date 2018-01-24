#coding: utf-8
from numba import jit
from pystats.utility import sum_of_square
import pandas as pd
import numpy as np

# 1 - way anova １要因被験者間分散分析
# 分散分析表をreturnする
@jit
def one_way_anova(df, factorCol, valCol):
    factors = list(pd.unique(df[factorCol]))
    factor_num = len(factors)
    data_num = df[valCol].count()

    mean_w = df[valCol].mean()
    mean_fs = {}
    deg_of_freedoms = {}
    for factor in factors:
        mean_fs[factor] = df[df[factorCol] == factor][valCol].mean()
        deg_of_freedoms[factor] = len(df[df[factorCol] == factor].values) - 1

    # Degree of freedoms
    dof_inter = factor_num - 1
    dof_intra = sum(deg_of_freedoms.values())
    dof_w = dof_inter + dof_intra

    # sum of squares
    sqrt_w = sum_of_square(df[valCol].values, mean_w)
    sqrt_intra = 0
    for factor in factors:
        sqrt_intra += sum_of_square(df[df[factorCol] == factor][valCol].values, mean_fs[factor])
    sqrt_inter = sqrt_w - sqrt_intra

    # mean squares
    ave_sqrt_inter = sqrt_inter / dof_inter
    ave_sqrt_intra = sqrt_intra / dof_intra

    F = ave_sqrt_inter / ave_sqrt_intra

    table = pd.DataFrame({
        'factor': ['inter', 'intra', 'whole'],
        'deg of freedom': [dof_inter, dof_intra, dof_w],
        'mean square': [ave_sqrt_inter, ave_sqrt_intra, None],
        'F': [F, None, None]
    })
    return table
