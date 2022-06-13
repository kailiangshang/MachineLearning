import numpy as np
from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.manifold import LocallyLinearEmbedding


class LLE:

    def __init__(self, X: ndarray, n_neighbors: int, n_components: int, dist_select='ecu'):
        """
        :param X: 原始数据集
        :param n_neighbors: k邻近阈值
        :param n_components: 降低到的维度数
        :param dist_select: 算法所根据的距离选择, 默认为欧氏距离
        """
        self.X = X
        self.k = n_neighbors
        self.d = n_components
        self.__epsilon = self.__get_epsilon()
        self._N, self._D = np.shape(X)  # 得到原始数据的维度

    def __get_epsilon(self):
        """
        :return: 给出一个阈值，防止由于邻近点过多导致降维归0的情况
        """
        if self.k > self.d:
            return 1e-3
        else:
            return 0

    def __euc_dist(self):
        """
        :return: 原始数据的欧式距离矩阵
        """
        dist_array = np.zeros([self._N, self._N])
        for i in range(self._N):
            for j in range(self._N):
                dist_array[i, j] = np.sqrt(np.dot((self.X[i] - self.X[j]), (self.X[i] - self.X[j]).T))
        return dist_array

    def __get_k_index(self):
        """
        :return: 获取每个样本点的k个临近点的位置索引
        """
        index = np.argsort(self.__euc_dist(), axis=1)[:, 1:self.k + 1]
        return index

    def __w_cal(self):
        """
        :return: 计算得到原始数据权重向量
        """
        w = np.zeros([self._N, self.k])
        index_NN = self.__get_k_index()
        I = np.ones([self.k, 1])  # 形成单位列向量
        for i in range(self._N):
            X_k = self.X[index_NN[i]]  # 得到数据的k个邻近点
            X_i = [self.X[i]]  # 得到原始数据

            S_i = np.dot((np.dot(I, X_i) - X_k), (np.dot(I, X_i) - X_k).T)
            # 防止对角线元素过小
            S_i = S_i + np.eye(self.k)*self.__epsilon*np.trace(S_i)
            S_i_inv = np.linalg.pinv(S_i)  # 计算矩阵的广义逆矩阵

            w[i] = np.dot(I.T, S_i_inv)/(np.dot(np.dot(I.T, S_i_inv), I))  # 计算原始权重
        return w

    def __W_cal(self):
        """
        :return: 得到权重系数矩阵
        """
        W = np.zeros([self._N, self._N])
        w = self.__w_cal()
        index = self.__get_k_index()
        for i in range(self._N):
            W[i, index[i]] = w[i]
        return W

    def fit(self):
        """
        :return: 得到降维后的数据结果
        """
        W = self.__W_cal()
        I_N = np.eye(self._N)
        C = np.dot((I_N-W).T, (I_N-W))

        # 特征值分解
        eig_val, eig_vector = np.linalg.eig(C)

        index = np.argsort(eig_val)[1:self.d+1]
        X_result = eig_vector[:, index]
        return X_result


if __name__ == '__main__':
    data = load_iris().data
    lle = LLE(data, n_neighbors=4, n_components=3)
    print(lle.fit())
    print(LocallyLinearEmbedding(n_neighbors=4, n_components=3).fit_transform(data))
