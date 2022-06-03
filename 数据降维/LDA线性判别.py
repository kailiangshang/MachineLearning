import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


class LDA:

    def __init__(self, X, y):
        """
        :param X: 原始数据集
        :param y: 原始标签,可以是字符串
        """
        self.X = X
        self.X_div = self.X.shape[-1]
        self.y = LabelEncoder().fit(y).transform(y)  # 对标签y进行编码
        self._class = np.unique(self.y)
        self._mean_vectors = self.__mean_vector()  # 不同类别的均值向量

    def __mean_vector(self):
        """
        :return: 计算所得各类别的均值向量
        """
        mean_vectors = []
        for class_ in self._class:
            mean_vectors.append(np.mean(self.X[self.y == class_], axis=0))
        return mean_vectors

    def __s_w(self):
        """
        :return: 类内散度矩阵
        """
        S_W = np.zeros((self.X_div, self.X_div))
        for class_, mean_vector in zip(self._class, self._mean_vectors):
            s_i = np.zeros((self.X_div, self.X_div))
            for x in self.X[self.y == class_]:
                x = x.reshape(self.X_div, 1)
                mean_vector = mean_vector.reshape(self.X_div, 1)
                s_i += (x-mean_vector).dot((x-mean_vector).T)
            S_W += s_i
        return S_W

    def __s_b(self):
        """
        :return: 计算类间散度矩阵
        """
        S_B = np.zeros((self.X_div, self.X_div))
        mean_vector_overall = np.mean(self.X, axis=0)
        for class_, mean_vector in zip(self._class, self._mean_vectors):
            N = self.X[self.y == class_, :].shape[0]
            mean_vector_overall = mean_vector_overall.reshape(self.X_div, 1)
            mean_vector = mean_vector.reshape(self.X_div, 1)
            S_B += N * (mean_vector - mean_vector_overall).dot((mean_vector - mean_vector_overall).T)
        return S_B

    def __eigen_(self):
        eig_value, eig_vector = np.linalg.eig(np.linalg.inv(self.__s_w()).dot(self.__s_b()))
        return eig_value, eig_vector

    def fit(self):
        _, eig_vector = self.__eigen_()
        return self.X.dot(eig_vector)


if __name__ == '__main__':
    data = load_iris().data
    y = load_iris().target
    lda = LDA(data, y)
    X = lda.fit()
    print(X)













