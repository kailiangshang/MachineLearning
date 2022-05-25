import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import neighbors

# 使用sklearn中自带的数据集
iris_data = datasets.load_iris()
X = iris_data.data[:, [0, 2]]
y = iris_data.target

# 绘图查看数据集
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.show()

# KNN模型拟合
# 要找到最好的n_neighbors 可以采用交叉验证的方式，计算模型得分，根据模型得分选择最合适的值
model_KNN = neighbors.KNeighborsClassifier(weights='distance', n_neighbors=20)
model_KNN.fit(X, y)
model_KNN.kneighbors_graph()


# 生成随机预测数据
x0_min, x0_max = X[:, 0].min()-1, X[:, 0].max()+1
x1_min, x1_max = X[:, 1].min()-1, X[:, 1].max()+1


# 生成网格数据并进行模型预测
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01),
                       np.arange(x1_min, x1_max, 0.01))
Z = model_KNN.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)

# 绘制非矩形网格彩色图
plt.pcolormesh(xx0, xx1, Z, shading='gouraud')
plt.colorbar()
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.show()
