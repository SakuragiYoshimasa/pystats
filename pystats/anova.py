#coding: utf-8
from numba import jit
from pystats.utility import sum_of_square
import pandas as pd
import numpy as np

# 1 - way anova １要因被験者間分散分析
# 分散分析表をreturnする
@jit
def one_way_anova(df, cohortCol, valCol):
    cohorts = list(pd.unique(df[cohortCol]))
    cohort_num = len(cohorts)
    data_num = df[valCol].count()

    mean_w = df[valCol].mean()
    mean_cs = {}
    deg_of_freedoms = {}
    for cohort in cohorts:
        mean_cs[cohort] = df[df[cohortCol] == cohort][valCol].mean()
        deg_of_freedoms[cohort] = len(df[df[cohortCol] == cohort].values) - 1

    # Degree of freedoms
    dof_inter = cohort_num - 1
    dof_intra = sum(deg_of_freedoms.values())
    dof_w = dof_inter + dof_intra

    # sum of squares
    sqrt_w = sum_of_square(df[valCol].values, mean_w)
    sqrt_intra = 0
    for cohort in cohorts:
        sqrt_intra += sum_of_square(df[df[cohortCol] == cohort][valCol].values, mean_cs[cohort])
    sqrt_inter = sqrt_w - sqrt_intra

    # mean squares
    ave_sqrt_inter = sqrt_inter / dof_inter
    ave_sqrt_intra = sqrt_intra / dof_intra

    F = ave_sqrt_inter / ave_sqrt_intra

    table = pd.DataFrame({
        'factor': ['inter', 'intra', 'whole'],
        'dof': [dof_inter, dof_intra, dof_w],
        'mean_S': [ave_sqrt_inter, ave_sqrt_intra, None],
        'F': [F, None, None]
    })
    return table.ix[:, ['factor', 'dof', 'mean_S', 'F']]
