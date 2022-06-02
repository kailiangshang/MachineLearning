import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np


class PCA:
    def __init__(self, X, n_feature=None):
        self.X = X
        self.n_feature = n_feature

    def __center_data(self):
        """
        :return: 归一化后的数据
        """
        self.X = self.X - self.X.mean(0)
        return self.X

    def x_cov(self):
        X = self.__center_data()
        X_cov = np.cov(X, rowvar=False)
        return X_cov

    def feature_decomposition(self):
        """
        :return: 计算协方差矩阵对应的特征值和特征向量
        """
        X_cov = self.x_cov()
        eigenvalue, feature_vector = np.linalg.eig(X_cov)
        return eigenvalue, feature_vector

    def w_select(self):
        """
        :return: 根据初始需要降维的数目形成特征向量矩阵，也就是X的投影矩阵,并获得投影后的X矩阵
        """
        eigenvalue, feature_vector = self.feature_decomposition()
        if self.n_feature is None:
            W = feature_vector
        else:
            W = feature_vector[:, :self.n_feature]
            eigenvalue = eigenvalue[:self.n_feature]
        X_projection = np.dot(self.__center_data(), W)
        return W, eigenvalue, X_projection

    def var_ratio(self):
        """
        :return: 计算方差贡献率.
        """
        _, eigenvalue, _ = self.w_select()
        var_ratio_ = [i / eigenvalue.sum() for i in eigenvalue]
        return var_ratio_


if __name__ == '__main__':
    data = load_iris().data
    pca = PCA(data)
    print(pca.w_select())
    print(pca.var_ratio())
    plt.plot(pca.var_ratio())
    plt.show()