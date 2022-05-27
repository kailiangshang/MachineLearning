import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import seaborn as sns


class SpectralC:

    def __init__(self, data, gamma=None, n_clusters=None):
        """
        :param data: 原始数据集
        :param gamma: 高斯核函数的参数
        :param n_clusters: 初始切图和最终切图的类别数目
        """
        self.X = data
        self.gamma = gamma
        self.n_clusters = n_clusters

    @staticmethod
    def get_gamma_k():
        """
        :return:不同量级下的gamma值
        """
        return [0.01, 0.1, 1, 10], [3, 4, 5, 6]

    @staticmethod
    def heat(df):
        df.index = [str(i) for i in df.index]
        df.columns = [str(i) for i in df.columns]
        return sns.heatmap(df, linecolor='k', annot=True)

    def score(self, y_pred):
        """
        :param y_pred: 模型预测分类结果
        :return: 模型分数值
        """
        score = metrics.calinski_harabasz_score(self.X, y_pred)
        return score

    def predict(self, gamma, k):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(self.X)
        return y_pred

    def get_info(self, gamma, k, info_df=None):
        for gamma_ in gamma:
            for k_ in k:
                y_pred = self.predict(gamma=gamma_, k=k_)
                score = self.score(y_pred)
                info_df.loc[gamma_, k_] = score
        return info_df

    def get_result(self):
        info_df = pd.DataFrame()
        if self.gamma is None:
            gamma, k = self.get_gamma_k()
        else:
            gamma = self.gamma
            k = self.n_clusters
        info_df = self.get_info(gamma=gamma, k=k, info_df=info_df)
        return info_df, self.heat(info_df)


if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=1000, n_features=3, centers=3, random_state=9)
    sp = SpectralC(data=X)
    info, pic = sp.get_result()
    print(info)
    plt.show()
