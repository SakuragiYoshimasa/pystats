#coding: utf-8
from numba import jit
from pystats.utility import sum_of_square
import pandas as pd
import numpy as np

# 1 - way anova １要因被験者間分散分析
# 分散分析表をreturnする
# Inter:=群間, Intra:=郡内
@jit
def one_way_anova_between_subject(df, levelCol, valCol):
    levels = list(pd.unique(df[levelCol]))
    level_num = len(levels)
    data_num = df[valCol].count()

    mean_w = df[valCol].mean()
    mean_ls = {}
    deg_of_freedoms = {}
    for level in levels:
        mean_ls[level] = df[df[levelCol] == level][valCol].mean()
        deg_of_freedoms[level] = len(df[df[levelCol] == level].values) - 1

    # Degree of freedoms
    dof_inter = level_num - 1
    dof_intra = sum(deg_of_freedoms.values())
    dof_w = dof_inter + dof_intra

    # sum of squares
    sqrt_w = sum_of_square(df[valCol].values, mean_w)
    sqrt_intra = 0
    for level in levels:
        sqrt_intra += sum_of_square(df[df[levelCol] == level][valCol].values, mean_ls[level])
    sqrt_inter = sqrt_w - sqrt_intra

    # mean squares
    ms_inter = sqrt_inter / dof_inter
    ms_intra = sqrt_intra / dof_intra

    F = ms_inter / ms_intra

    table = pd.DataFrame({
        'factor': ['inter', 'intra', 'whole'],
        'dof': [dof_inter, dof_intra, dof_w],
        'mean_S': [ms_inter, ms_intra, None],
        'F': [F, None, None]
    })
    return table.ix[:, ['factor', 'dof', 'mean_S', 'F']]

@jit
def one_way_anova_within_subject(df, levelCol, subjectCol, valCol):
    levels = list(pd.unique(df[levelCol]))
    level_num = len(levels)
    data_num = df[valCol].count()

    mean_w = df[valCol].mean()
    mean_ls = {}
    deg_of_freedoms = {}
    for level in levels:
        mean_ls[level] = df[df[levelCol] == level][valCol].mean()
        deg_of_freedoms[level] = len(df[df[levelCol] == level].values) - 1

    # Degree of freedoms
    dof_cond = level_num - 1
    dof_error = sum(deg_of_freedoms.values())
    dof_w = dof_cond + dof_error

    # sum of squares
    sos_w = sum_of_square(df[valCol].values, mean_w)
    sos_error = 0 # 誤差平方和
    for level in levels:
        sos_error += sum_of_square(df[df[levelCol] == level][valCol].values, mean_ls[level])
    sos_cond = sos_w - sos_error # 条件平方和

    # mean squares
    ms_cond = sos_cond / dof_cond
    ms_error = sos_error / dof_error

    F = ms_cond / ms_error

    # 個人差による平方和
    subjects = list(pd.unique(df[subjectCol]))
    dof_subject = len(subjects) - 1 # 個人差の自由度
    mean_ss = {} # 各被験者の平均
    for subject in subjects:
        mean_ss[subject] = df[df[subjectCol] == subject][valCol].mean()

    # 個人差の平方和
    sos_subjects = 0
    for subject in subjects:
        sos_subjects += np.square(mean_ss[subject] - mean_w) * len(df[df[subjectCol] == subject])

    # 残差
    residual_error = sos_error - sos_subjects
    dof_residual_e = dof_w - dof_cond - dof_subject
    ms_subject = sos_subjects / dof_subject
    ms_residual_e = residual_error / dof_residual_e

    F = ms_cond / ms_residual_e

    table = pd.DataFrame({
        'factor': ['condition', 'Individual_diff', 'residual_error', 'whole'],
        'dof': [dof_cond, dof_subject, dof_residual_e, dof_w],
        'mean_S': [ms_cond, ms_subject, ms_residual_e, None],
        'sos': [sos_cond, sos_subjects, residual_error, sos_w],
        'F': [F, None, None, None]
    })
    return table.ix[:, ['factor', 'dof', 'sos', 'mean_S', 'F']]
