#coding: utf-8
from pystats.anova import one_way_anova_between_subject
from pystats.anova import one_way_anova_within_subject
from pystats.anova import two_way_anova_between_subject
from pystats.anova import two_way_anova_within_subject
from pystats.anova import two_way_anova_within_subject_jit
import pandas as pd
from numba import jit
'''
sample_data = {
    'type': ['A','A','A','A','B','B','B','B','C','C','C','C',],
    'value': [3,4,4,3,7,8,9,6,5,4,6,7] }
print(one_way_anova_between_subject(pd.DataFrame(sample_data), levelCol='type', valCol='value'))

sample_data = {
    'type': ['A','A','A','A','B','B','B','B','C','C','C','C',],
    'subject': ['1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3', '4'],
    'value': [10,9,4,7,5,4,2,3,9,5,3,5] }
print(one_way_anova_within_subject(pd.DataFrame(sample_data), levelCol='type', subjectCol='subject', valCol='value'))
'''
'''
sample_data = pd.DataFrame({
    'levelA': ['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2'],
    'levelB': ['1','1','1','1','1','2','2','2','2','2','3','3','3','3','3','1','1','1','1','1','2','2','2','2','2','3','3','3','3','3'],
    'value':  [6,4,5,3,2,10,8,10,8,9,11,12,12,10,10,5,4,2,2,2,7,6,5,4,3,12,8,5,6,4] })
print(two_way_anova_between_subject(sample_data, levelACol='levelA', levelBCol='levelB', valCol='value'))
'''

sample_data = {
    'levelA': ['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2'],
    'levelB': ['1','1','1','1','1','2','2','2','2','2','3','3','3','3','3','1','1','1','1','1','2','2','2','2','2','3','3','3','3','3'],
    'subject':['1','2','3','4','5','1','2','3','4','5','1','2','3','4','5','1','2','3','4','5','1','2','3','4','5','1','2','3','4','5'],
    'value':  [6.0,4.0,5.0,3.0,2.0,10.0,8.0,10.0,8.0,9.0,11.0,12.0,12.0,10.0,10.0,5.0,4.0,2.0,2.0,2.0,7.0,6.0,5.0,4.0,3.0,12.0,8.0,5.0,6.0,4.0] }
#print(two_way_anova_within_subject(pd.DataFrame(sample_data), levelACol='levelA', levelBCol='levelB', subjectCol='subject', valCol='value'))

import time

@jit
def jited():
    start = time.time()
    for i in range(100):
        two_way_anova_within_subject_jit(pd.DataFrame(sample_data), levelACol='levelA', levelBCol='levelB', subjectCol='subject', valCol='value')
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

def nojited():
    start = time.time()
    for i in range(100):
        two_way_anova_within_subject(pd.DataFrame(sample_data), levelACol='levelA', levelBCol='levelB', subjectCol='subject', valCol='value')
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == '__main__':
    jited()
    nojited()
