import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import datasets
from sklearn import tree
from graphviz import Digraph
# 决策树有两种类型,一种是分类树,一种是回归树.
# 分类树用来预测类别,回归树返回一个值

# 用sklearn自带的数据集
iris_data = datasets.load_iris()
print(iris_data)
X = iris_data.data[:, [0, 1, 2]]
print(X)
y = iris_data.target

# 训练模型,可以限制树的最大的深度
model_class = tree.DecisionTreeClassifier(max_depth=4)
model_class.fit(X, y)


# 绘制三维轮廓图
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
z_min, z_max = X[:, 2].min()-1, X[:, 2].max()+1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1),
                         np.arange(z_min, z_max, 0.1))
print(xx, yy)
Z = model_class.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)
fig = plt.figure()
# ax = fig.gca(projection='3d')
plt.scatter(xx, yy, zz)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.show()


# 绘制树图
plt.figure(dpi=500)
tree.plot_tree(model_class, class_names=iris_data.target_names,
               feature_names=iris_data.feature_names, filled=True,
               rounded=True)
plt.show()






























