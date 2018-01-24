#coding: utf-8
from pystats.anova import one_way_anova_between_subject
from pystats.anova import one_way_anova_within_subject
import pandas as pd

sample_data = {
    'type': ['A','A','A','A','B','B','B','B','C','C','C','C',],
    'value': [3,4,4,3,7,8,9,6,5,4,6,7] }
print(one_way_anova_between_subject(pd.DataFrame(sample_data), levelCol='type', valCol='value'))

sample_data = {
    'type': ['A','A','A','A','B','B','B','B','C','C','C','C',],
    'subject': ['1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3', '4'],
    'value': [10,9,4,7,5,4,2,3,9,5,3,5] }
print(one_way_anova_within_subject(pd.DataFrame(sample_data), levelCol='type', subjectCol='subject', valCol='value'))
