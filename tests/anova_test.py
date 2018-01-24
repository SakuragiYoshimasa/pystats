#coding: utf-8
from pystats.anova import one_way_anova
import pandas as pd

sample_data = {
    'type': ['A','A','A','A','B','B','B','B','C','C','C','C',],
    'value': [3,4,4,3,7,8,9,6,5,4,6,7] }
print(one_way_anova(pd.DataFrame(sample_data), factorCol='type', valCol='value'))
