from sklearn import datasets
from collections import defaultdict
import numpy as np
y = defaultdict(lambda: 0)
print(y)
X = datasets.load_iris().data
for xx in X:
    xx = tuple(xx)
    print(xx)
    y[xx] += 1.0 / X.shape[0]
print(y)


