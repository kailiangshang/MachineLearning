from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn import tree
# 应用回归树的前提是标签的输出值是连续的，不能是简单的分类问题

# sklearn 数据集
iris_data = datasets.load_iris()
X = iris_data.data
y = iris_data.target

# 训练模型
model_reg = DecisionTreeRegressor()
model_reg.fit(X, y)


# 决策树绘图
plt.figure(dpi=500)
tree.plot_tree(model_reg, class_names=iris_data.target_names,
               feature_names=iris_data.feature_names, filled=True,
               rounded=True)
plt.show()
















