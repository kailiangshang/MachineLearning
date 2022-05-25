"""
应用sklearn中的鸢尾花数据集特征的其中两列，实现SVM RBF
支持向量机（RBF高斯核函数）
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# 生成数据集,选取特征的第0列和第3列
data = datasets.load_iris()
X = StandardScaler().fit_transform(X=data.data[:, [0, 2]])
y = data.target
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.show()

# LinerSVC模型训练（ovo可以提升训练的精确度）
L_model = LinearSVC(multi_class='ovr', random_state=42)
L_model.fit(X, y)

# SVC模型训练(包括参数选择和模型训练，其中C表示惩罚项系数，gamma表示单个样本对超平面分类的影响）
param_dict = dict(C=[0.1, 1, 10], gamma=[1, 0.1, 0.01])
gird = GridSearchCV(SVC(), param_grid=param_dict, cv=10)
gird.fit(X, y)

# 两个模型对比，参数对比和得分对比
result_compare = pd.DataFrame(
    {
        'score': [L_model.score(X, y), gird.best_score_],
        'param': [L_model.get_params(), gird.best_params_]
    }, index=['L_SVC', 'SVC(RBF)']
)
print(result_compare)

# 绘图查看两个模型的分类结果
x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_min, x1_max = X[:, 1].min() - 1, X[:, 0].max() + 1
x0, x1 = np.meshgrid(
    np.arange(x0_min, x0_max, 0.02),
    np.arange(x1_min, x1_max, 0.02)
)


def L_graph():
    """
    params: None
    return: None
    Note: 输出L_model模型的轮廓图
    """
    global L_model, x0, x1
    Z = L_model.predict(np.c_[x0.ravel(), x1.ravel()])
    Z = Z.reshape(x0.shape)
    plt.contourf(x0, x1, Z, alpha=0.8, )
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.show()


def SVC_graph():
    for i, C in enumerate((0.1, 1, 10)):
        for j, gamma in enumerate((1, 0.1, 0.01)):
            plt.subplot()
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X, y)
            Z = clf.predict(np.c_[x0.ravel(), x1.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(x0.shape)
            plt.contourf(x0, x1, Z, alpha=0.8)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

            plt.xlim(x0.min(), x1.max())
            plt.ylim(x1.min(), x1.max())
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(" gamma=" + str(gamma) + " C=" + str(C))
            plt.show()


if __name__ == '__main__':
    L_graph()
    SVC_graph()
