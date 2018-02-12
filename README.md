# pystats

Python package for data analysis.
Now in development.

## Installation

```
python setup.py install --record files.txt
```

### Unsntall

```
cat files.txt | xargs rm -rf
```

### Dependency
- numpy
- pandas
- cython

## Usage

```python
from pystats.anova import one_way_anova_between_subject
from pystats.anova import one_way_anova_within_subject
from pystats.anova import two_way_anova_between_subject
from pystats.anova import two_way_anova_within_subject
import pandas as pd

sample_data = {
    'levelA': ['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2'],
    'levelB': ['1','1','1','1','1','2','2','2','2','2','3','3','3','3','3','1','1','1','1','1','2','2','2','2','2','3','3','3','3','3'],
    'subject':['1','2','3','4','5','1','2','3','4','5','1','2','3','4','5','1','2','3','4','5','1','2','3','4','5','1','2','3','4','5'],
    'value':  [6.0,4.0,5.0,3.0,2.0,10.0,8.0,10.0,8.0,9.0,11.0,12.0,12.0,10.0,10.0,5.0,4.0,2.0,2.0,2.0,7.0,6.0,5.0,4.0,3.0,12.0,8.0,5.0,6.0,4.0] }
print(two_way_anova_within_subject(pd.DataFrame(sample_data), levelACol='levelA', levelBCol='levelB', subjectCol='subject', valCol='value')) # return pandas dataframe of anova table.
```

or there is fast version for only 'two way anova within subject'.
```python

from pystats.anova import twawis_mat
sample_data = np.array([[6.0,4.0,5.0,3.0,2.0], [10.0,8.0,10.0,8.0,9.0], [11.0,12.0,12.0,10.0,10.0], [5.0,4.0,2.0,2.0,2.0], [7.0,6.0,5.0,4.0,3.0], [12.0,8.0,5.0,6.0,4.0]])
print(twawis_mat(sample_data, 2, 3)) # return F_A, F_B, F_interation
```
