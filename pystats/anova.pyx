#coding: utf-8

import pandas as pd
import numpy as np
cimport numpy as np

def sum_of_square(np.ndarray[double, ndim=1] data, double mean):
    return sum(np.square(data - mean))

def sum_of_square_with_weight(np.ndarray[double, ndim=1] data, double mean, np.ndarray[double, ndim=1] weight):
    return sum(np.square(data - mean) * weight)

def two_way_anova_between_subject(df, char* levelACol, char* levelBCol, char* valCol):
    levelAs = list(pd.unique(df[levelACol]))
    levelBs = list(pd.unique(df[levelBCol]))

    #セル平均
    cell_mean = {}
    cell_mean['mean_B'] = {}
    for levelA in levelAs:
        cell_mean[levelA] = {}
        for levelB in levelBs:
            cell_mean[levelA][levelB] = df[(df[levelACol] == levelA) & (df[levelBCol] == levelB)][valCol].mean()
            if levelB not in cell_mean['mean_B'].keys():
                cell_mean['mean_B'][levelB] = df[df[levelBCol] == levelB][valCol].mean()
        cell_mean[levelA]['mean_A'] = df[df[levelACol] == levelA][valCol].mean()
    cell_mean['whole'] = df[valCol].mean()

    # 全体平方和
    cdef double sos_w = sum_of_square(df[valCol].values, cell_mean['whole'])

    # 要因Aの主効果の平方和
    cdef double sos_factorA = 0
    for levelA in levelAs:
        sos_factorA += np.square(cell_mean[levelA]['mean_A'] - cell_mean['whole']) * len(df[df[levelACol] == levelA].values)

    # 要因Bの主効果の平方和
    cdef double sos_factorB = 0
    for levelB in levelBs:
        sos_factorB += np.square(cell_mean['mean_B'][levelB] - cell_mean['whole']) * len(df[df[levelBCol] == levelB].values)

    # セル平均の平方和
    cdef double sos_cell = 0
    for levelA in levelAs:
        for levelB in levelBs:
            sos_cell += np.square(cell_mean[levelA][levelB] - cell_mean['whole']) * len(df[(df[levelACol] == levelA) & (df[levelBCol] == levelB)].values)

    # 交互作用の平方和
    cdef double sos_interaction = sos_cell - sos_factorA - sos_factorB

    # 誤差の平方和
    cdef double sos_error = 0
    for levelA in levelAs:
        for levelB in levelBs:
            sos_error += sum_of_square(df[(df[levelACol] == levelA) & (df[levelBCol] == levelB)][valCol].values, cell_mean[levelA][levelB])

    # 自由度
    cdef double dof_w = len(df[valCol].values) - 1
    cdef double dof_factorA = len(levelAs) - 1
    cdef double dof_factorB = len(levelBs) - 1
    cdef double dof_interaction = dof_factorA * dof_factorB
    cdef double dof_error = dof_w - dof_factorA - dof_factorB - dof_interaction

    cdef double ms_factorA = sos_factorA / dof_factorA
    cdef double ms_factorB = sos_factorB / dof_factorB
    cdef double ms_interation = sos_interaction / dof_interaction
    cdef double ms_error = sos_error / dof_error

    cdef double F_A = ms_factorA / ms_error
    cdef double F_B = ms_factorB / ms_error
    cdef double F_interation = ms_interation / ms_error

    table = pd.DataFrame({
        'factor': [levelACol, levelBCol, 'interaction', 'error', 'whole'],
        'sos': [sos_factorA, sos_factorB, sos_interaction, sos_error, sos_w],
        'dof': [dof_factorA, dof_factorB, dof_interaction, dof_error, dof_w],
        'mean_S': [ms_factorA, ms_factorB, ms_interation, ms_error, None],
        'F': [F_A, F_B, F_interation, None, None]
    })
    return table.ix[:, ['factor', 'sos', 'dof', 'mean_S', 'F']]

def twawis_mat(np.ndarray[double, ndim=2] data, int levelANum, int levelBNum):

  cdef double w_mean = np.mean(data)
  cell_means_levelA_levelB = np.mean(data, axis=1)
  cell_means_levelAs = np.array([np.mean(data[i * levelBNum: (i + 1) * levelBNum]) for i in range(levelANum)])
  cell_means_levelBs = np.array([np.mean(data[i::levelBNum]) for i in range(levelBNum)])

  cdef int subjectCount = len(data[0])
  # 全体平方和
  cdef double sos_w = sum_of_square(np.ravel(data), w_mean)
  # 要因Aの主効果の平方和
  cdef double sos_factorA = sum_of_square(cell_means_levelAs, w_mean) * levelBNum * subjectCount
  # 要因Bの主効果の平方和
  cdef double sos_factorB = sum_of_square(cell_means_levelBs, w_mean) * levelANum * subjectCount
  # セル平均の平方和
  cdef double sos_cell = sum_of_square(cell_means_levelA_levelB, w_mean) * subjectCount
  # 交互作用の平方和
  cdef double sos_interaction = sos_cell - sos_factorA - sos_factorB
  # 誤差の平方和
  cdef int dataSize = len(data)
  errors = [sum_of_square(data[i], cell_means_levelA_levelB[i]) for i in range(dataSize)]
  cdef double sos_error = np.sum(errors)

  # ここまでは同様
  # 誤差の平方和 = 個人差の平方和 + 要因Aに対する誤差の平方和 + 要因Bに対する誤差の平方和 + 交互作用に対する誤差の平方和　に分解
  mean_ss = np.mean(data, axis=0) # 各被験者の平均
  # 個人差の平方和
  cdef double sos_subject = sum_of_square(mean_ss, w_mean) * dataSize
  # 要因Aに対する誤差の平方和
  cdef double sos_factorA_error = sum_of_square(np.mean(np.reshape(data, (levelBNum, subjectCount * levelANum), 'F'), axis=0), w_mean) * levelBNum
  sos_factorA_error -= sos_factorA + sos_subject
  # 要因Bに対する誤差の平方和
  cdef double sos_factorB_error = sum_of_square(np.mean(np.reshape(data, (levelANum, subjectCount * levelBNum)), axis=0), w_mean) * levelANum
  sos_factorB_error -= sos_factorB + sos_subject
  # 交互作用に対する誤差の平均和
  cdef double sos_interaction_error = sos_error - sos_subject - sos_factorA_error - sos_factorB_error
  # 自由度
  cdef double dof_w = levelANum * levelBNum * subjectCount - 1
  cdef double dof_factorA = levelANum - 1
  cdef double dof_factorB = levelBNum - 1
  cdef double dof_interaction = dof_factorA * dof_factorB
  cdef double dof_subject = subjectCount - 1
  cdef double dof_AxS = dof_factorA * dof_subject
  cdef double dof_BxS = dof_factorB * dof_subject
  cdef double dof_AxBxS = dof_interaction * dof_subject

  cdef double ms_factorA = sos_factorA / dof_factorA
  cdef double ms_factorB = sos_factorB / dof_factorB
  cdef double ms_interation = sos_interaction / dof_interaction
  cdef double ms_subject = sos_subject / dof_subject
  cdef double ms_AxS = sos_factorA_error / dof_AxS
  cdef double ms_BxS = sos_factorB_error / dof_BxS
  cdef double ms_AxBxS = sos_interaction_error / dof_AxBxS

  cdef double F_A = ms_factorA / ms_AxS
  cdef double F_B = ms_factorB / ms_BxS
  cdef double F_interation = ms_interation / ms_AxBxS

  return [F_A, F_B, F_interation]

def two_way_anova_within_subject(df, str levelACol, str levelBCol, str subjectCol, str valCol):

    levelAs = list(pd.unique(df[levelACol]))
    levelBs = list(pd.unique(df[levelBCol]))

    #セル平均
    cell_mean = {}
    cell_mean['mean_B'] = {}
    for levelA in levelAs:
        cell_mean[levelA] = {}
        for levelB in levelBs:
            cell_mean[levelA][levelB] = df[(df[levelACol] == levelA) & (df[levelBCol] == levelB)][valCol].mean()
            if levelB not in cell_mean['mean_B'].keys():
                cell_mean['mean_B'][levelB] = df[df[levelBCol] == levelB][valCol].mean()
        cell_mean[levelA]['mean_A'] = df[df[levelACol] == levelA][valCol].mean()
    cell_mean['whole'] = df[valCol].mean()

    # 全体平方和
    cdef double sos_w = sum_of_square(df[valCol].values, cell_mean['whole'])
    # 要因Aの主効果の平方和
    cdef double sos_factorA = sum_of_square_with_weight(np.array([cell_mean[levelA]['mean_A'] for levelA in levelAs], dtype='float64'), cell_mean['whole'], np.array([len(df[df[levelACol] == levelA].values) for levelA in levelAs], dtype='float64'))
    # 要因Bの主効果の平方和
    cdef double sos_factorB = sum_of_square_with_weight(np.array([cell_mean['mean_B'][levelB] for levelB in levelBs], dtype='float64'), cell_mean['whole'], np.array([len(df[df[levelBCol] == levelB].values) for levelB in levelBs], dtype='float64'))
    # セル平均の平方和
    cdef double sos_cell = sum_of_square_with_weight(np.array([cell_mean[levelA][levelB] for levelA in levelAs for levelB in levelBs], dtype='float64'), cell_mean['whole'], np.array([len(df[(df[levelACol] == levelA) & (df[levelBCol] == levelB)].values) for levelA in levelAs for levelB in levelBs], dtype='float64'))
    # 交互作用の平方和
    cdef double sos_interaction = sos_cell - sos_factorA - sos_factorB
    # 誤差の平方和
    cdef double sos_error = np.sum(np.array([sum_of_square(df[(df[levelACol] == levelA) & (df[levelBCol] == levelB)][valCol].values, cell_mean[levelA][levelB]) for levelA in levelAs for levelB in levelBs], dtype='float64'))

    # ここまでは同様
    # ここからが圧倒的に遅い
    # 誤差の平方和 = 個人差の平方和 + 要因Aに対する誤差の平方和 + 要因Bに対する誤差の平方和 + 交互作用に対する誤差の平方和　に分解
    subjects = list(pd.unique(df[subjectCol]))
    mean_ss = {subject: df[df[subjectCol] == subject][valCol].mean() for subject in subjects} # 各被験者の平均

    # 個人差の平方和
    cdef double sos_subject = sum_of_square_with_weight(np.array(list(mean_ss.values()), dtype='float64'), cell_mean['whole'], np.array([len(df[df[subjectCol] == subject][valCol].values) for subject in subjects], dtype='float64'))
    # 要因Aに対する誤差の平方和
    cdef double sos_factorA_error = sum_of_square_with_weight(np.array([df[(df[subjectCol] == subject) & (df[levelACol] == levelA)][valCol].mean() for levelA in levelAs for subject in subjects], dtype='float64'), cell_mean['whole'], np.array([len(df[(df[subjectCol] == subject) & (df[levelACol] == levelA)][valCol].values) for levelA in levelAs for subject in subjects], dtype='float64'))
    sos_factorA_error -= sos_factorA + sos_subject
    # 要因Bに対する誤差の平方和
    #sos_factorB_error = 0
    cdef double sos_factorB_error = sum_of_square_with_weight(np.array([df[(df[subjectCol] == subject) & (df[levelBCol] == levelB)][valCol].mean() for levelB in levelBs for subject in subjects], dtype='float64'), cell_mean['whole'], np.array([len(df[(df[subjectCol] == subject) & (df[levelBCol] == levelB)][valCol].values) for levelB in levelBs for subject in subjects], dtype='float64'))
    sos_factorB_error -= sos_factorB + sos_subject
    # 交互作用に対する誤差の平均和
    cdef double sos_interaction_error = sos_error - sos_subject - sos_factorA_error - sos_factorB_error

    # 自由度
    cdef double dof_w = len(df[valCol].values) - 1
    cdef double dof_factorA = len(levelAs) - 1
    cdef double dof_factorB = len(levelBs) - 1
    cdef double dof_interaction = dof_factorA * dof_factorB
    cdef double dof_subject = len(subjects) - 1
    cdef double dof_AxS = dof_factorA * dof_subject
    cdef double dof_BxS = dof_factorB * dof_subject
    cdef double dof_AxBxS = dof_interaction * dof_subject

    cdef double ms_factorA = sos_factorA / dof_factorA
    cdef double ms_factorB = sos_factorB / dof_factorB
    cdef double ms_interation = sos_interaction / dof_interaction
    cdef double ms_subject = sos_subject / dof_subject
    cdef double ms_AxS = sos_factorA_error / dof_AxS
    cdef double ms_BxS = sos_factorB_error / dof_BxS
    cdef double ms_AxBxS = sos_interaction_error / dof_AxBxS

    cdef double F_A = ms_factorA / ms_AxS
    cdef double F_B = ms_factorB / ms_BxS
    cdef double F_interation = ms_interation / ms_AxBxS

    table = pd.DataFrame({
        'factor': ['subject',levelACol, 'A x S', levelBCol, 'B x S', 'interaction', 'interaction x S', 'whole'],
        'sos': [sos_subject, sos_factorA, sos_factorA_error, sos_factorB, sos_factorB_error, sos_interaction, sos_interaction_error, sos_w],
        'dof': [dof_subject, dof_factorA, dof_AxS, dof_factorB, dof_BxS, dof_interaction, dof_AxBxS, dof_w],
        'mean_S': [ms_subject, ms_factorA, ms_AxS, ms_factorB, ms_BxS, ms_interation, ms_AxBxS, None],
        'F': [None, F_A, None, F_B, None, F_interation, None, None]
    })
    return table.ix[:, ['factor', 'sos', 'dof', 'mean_S', 'F']]

# 1 - way anova １要因被験者間分散分析
# 分散分析表をreturnする
# Inter:=群間, Intra:=郡内

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

# 1 - way anova １要因被験者内分散分析
# 分散分析表をreturnする
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
