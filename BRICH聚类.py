import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
import seaborn as sns


class Birch_cluster:

    def __init__(self, data_X, B=None, T=None, k=None):
        self.X = self.Scaler_one(data_X)  # 原始数据
        self.B = B  # 内部节点最大CF数
        self.T = T  # 超球体半径
        self.k = k  # 先验分类类别数

    @staticmethod
    def Scaler_one(data_):
        """
        对原始数据集进行特征归一化处理
        """
        data_ = MinMaxScaler().fit_transform(data_)
        return data_

    @staticmethod
    def score(data_, y_pred):
        """
        对聚类结果进行评分
        """
        return metrics.calinski_harabasz_score(data_, y_pred)

    def plot(self, y=None, t=None, b=None, k=None):
        """
        对数据进行绘图，为了简化数据维度默认为2维
        """
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y, edgecolors='k')
        plt.title(f'T={t}, B={b}, k={k}')
        plt.show()

    def B_T(self):
        """
        如果初始传入中没有给定B和T的值，那么采用此值进行传输
        但是要注意使用此处的值，要对数据进行归一化处理
        生成BT值组成的矩阵，并绘制两者分数的热力图
        """
        B = list(int(i) for i in range(2, 32))
        T = np.linspace(0, self.X.shape[1], 21)
        b_t0, b_t1 = np.meshgrid(B, T)
        return np.c_[b_t0.ravel(), b_t1.ravel()][:, 0], np.c_[b_t0.ravel(), b_t1.ravel()][:, 1]

    @property
    def y_pred(self):
        score = []
        category = []
        for B in list(int(i) for i in range(2, 32)):
            for T in np.linspace(0, self.X.shape[1], 21):
                try:
                    result_y_pred = Birch(threshold=T, branching_factor=B).fit_predict(self.X)
                    score_ = self.score(self.X, result_y_pred)
                    score.append(score_)
                    category.append(result_y_pred.max() + 1)
                except:
                    score.append(0.000)
                    category.append(0.000)
        return category, score

    def run(self):
        info_df = pd.DataFrame()
        info_df['B'], info_df['T'] = self.B_T()
        info_df['B'].apply(lambda x: int(x))
        info_df['category'], info_df['score'] = self.y_pred
        info_plot = pd.DataFrame(index=info_df['B'].unique(), columns=info_df['T'].unique())
        for i, j, score in zip(info_df['B'], info_df['T'], info_df['score']):
            if score is not np.nan:
                info_plot.loc[i, j] = score
            else:
                info_plot.loc[i, j] = 0
        return info_df, info_plot


if __name__ == '__main__':
    data = load_iris().data[:, [0, 1]]
    result = Birch_cluster(data_X=data)
    result0, result1 = result.run()
    print(result1)
    plt.figure(dpi=10000)
    sns.heatmap(result1)
    plt.show()
