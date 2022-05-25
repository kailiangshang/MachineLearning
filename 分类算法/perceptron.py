import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

# 数据标签标准化
data = pd.read_csv('D:/pr学习/MachineLeaning/ML03_分类问题/01_LogisticRegression-master/data2.csv', index_col=0)
data['label'][data['label'] == 0] = -1
print(data.head())

# 模型训练
model = Perceptron(alpha=0.0000001)
model.fit(data[['grade1', 'grade2']], data['label'])
w = model.coef_
print(w)
b = model.intercept_
print(b)
for x in np.linspace(-1, 1, 1000):
    y = -w[0][0]*x/w[0][1]
    plt.scatter(x, y, c='k')


# 绘图查看
plt.scatter('grade1', 'grade2', data=data[data['label'] == 1])
plt.scatter('grade1', 'grade2', data=data[data['label'] == -1])
plt.show()




















