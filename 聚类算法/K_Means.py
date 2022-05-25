import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import metrics


class K_means:

    def __init__(self, data_X, k_list=None):
        """
        params:
        data_X  传入数据为np.array,
        data_y  为列表
        k_list  列表
        return: None
        """
        self.k = k_list
        self.X = data_X
        self.div = self.X.shape[1]

    def km_model(self, k_):
        km_ = KMeans(n_clusters=k_, random_state=42)
        y_pred = km_.fit_predict(self.X)
        return y_pred

    def plot(self, data_, y=None, k=None, score=None):
        if self.div == 2:
            plt.scatter(data_[:, 0], data_[:, 1], c=y, edgecolors='k')
            plt.title(f'k={k}, score={score}')
            plt.show()
        elif self.div == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2], c=y, edgecolors='k')
            ax.set_title(f'k={k}, score={score}')
            plt.show()
        else:
            print(f'数据为{self.div}维，超过3维，无法可视化')

    def k_select(self):
        k_result = {}
        self.plot(self.X)
        if self.k is None:
            param_dict = {'n_clusters': [i for i in range(0, 10)]}
            k_cv = GridSearchCV(KMeans(), param_grid=param_dict, cv=10)
            k_cv.fit(self.X)
            self.k = k_cv.best_params_['n_clusters']
            y = self.km_model(k_=self.k)
            score = metrics.calinski_harabasz_score(self.X, y)
            self.plot(self.X, y=y, score=score, k=self.k)
        else:
            for k_ in self.k:
                y = self.km_model(k_=k_)
                score = metrics.calinski_harabasz_score(self.X, y)
                self.plot(self.X, y=y, k=k_, score=score)


if __name__ == '__main__':
    data = load_iris().data[:, 0:3]
    km = K_means(data_X=data, k_list=[2, 3, 4, 5])
    km.k_select()

