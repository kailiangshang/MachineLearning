import numpy as np
import pandas as pd
from sklearn import datasets
from K_Means import *
from sklearn.metrics import homogeneity_score
from sklearn.cluster import DBSCAN
import seaborn as sns
from numpy import unique


class DBS:
    def __init__(self, data_X, y_true=None, eps=None, min_sample=None):
        """
        :param data_X: 原始数据集
        :param y_true: 模型真实分类结果，传入为列表
        :param eps: 邻域阈值，传入为列表
        :param min_sample: 邻域内样本数阈值，传入为列表
        注意：这里绘图并没有绘制所有的图，这是有限制的，热力图的绘制没有显示使译文图例过多导致的
        """
        self.data = data_X
        self.y = y_true
        self.eps = eps
        self.min_sample = min_sample

    def predict(self, eps=None, min_sample=None):
        """
        :param eps: 核心对象的邻域阈值
        :param min_sample: 核心对象邻域内的最小样本点数
        :return: 模型预测分类
        """
        y_pred = DBSCAN(eps=float(eps), min_samples=float(min_sample)).fit_predict(self.data)
        return y_pred

    def score(self, y_pred):
        score = homogeneity_score(self.y, y_pred)
        return score

    @staticmethod
    def plot(data_, c=None, eps=None, min_samples=None, k=None, score=None):
        """
        :param data_: 原始数据集，要求为ndarray对象，且数据集维度为二维
        :param c: 颜色列表
        :param eps: 核心对象的邻域阈值
        :param min_samples:核心对象邻域内的最小样本点数
        :param k:分类类别数目
        :param score：该条件下模型的得分homogeneity_score
        :return:各种情况下的绘图展示分类结果
        """

        fig1 = plt.figure(1)
        ax = fig1.add_subplot(111)
        plt.title(f'eps={eps}, min_samples={min_samples}, k_result={k}, score={score}')
        ax.scatter(data_[:, 0], data_[:, 1], c=c, edgecolors='k')
        plt.show()


    @staticmethod
    def eps_min():
        eps_list = np.linspace(0.1, 1, 10)
        min_sample_list = range(0, 11)
        eps_list, min_sample_list = np.meshgrid(eps_list, min_sample_list)
        return np.c_[eps_list.ravel(), min_sample_list.ravel()][:, 0], np.c_[eps_list.ravel(), min_sample_list.ravel()][:, 1]

    @staticmethod
    def heat(data_):
        """
        :param data_: 传入数据为DF对象，记得索引要为字符串
        :return: 分数热力图
        """
        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111)
        sns.heatmap(data_, linecolor='k', annot=True, ax=ax)
        plt.show()

    def get_result(self):
        score_list = []
        result_df = pd.DataFrame()

        if self.eps is None:
            eps_list, min_sample_list = self.eps_min()
            info_df = pd.DataFrame()
            result_df['eps'] = eps_list
            result_df['min_sample'] = min_sample_list
            for eps, min_sample in zip(eps_list, min_sample_list):
                eps = str(eps)
                min_sample = str(min_sample)
                y_pred = self.predict(eps, min_sample)
                score = self.score(y_pred)
                score_list.append(score)
                info_df.loc[eps, min_sample] = score
                self.plot(self.data, c=y_pred, eps=eps, min_samples=min_sample, k=y_pred.max() + 1, score=score)

        else:
            result_df['eps'] = self.eps
            result_df['min_sample'] = self.min_sample
            info_df = pd.DataFrame()
            for eps, min_sample in zip(self.eps, self.min_sample):
                y_pred = self.predict(eps, min_sample)
                score = self.score(y_pred)
                score_list.append(score)
                info_df.loc[eps, min_sample] = score
                self.plot(self.data, c=y_pred, eps=eps, min_samples=min_sample, k=y_pred.max() + 1, score=score)

        self.heat(info_df)



        result_df['score'] = score_list
        return result_df


if __name__ == '__main__':
    # 生成数据集
    X1, y1 = datasets.make_circles(n_samples=3000, factor=0.6, noise=0.05)
    X2, y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=[[1.2, 1.2]], random_state=9, cluster_std=[0.1])
    y2 = [i+2 for i in y2]
    # 测试数据集函数
    dbs = DBS(data_X=np.concatenate((X1, X2)), y_true=np.concatenate((y1, y2)))
    result = dbs.get_result()



