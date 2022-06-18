import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn import tree
from sklearn.datasets import load_iris


class Ada_CART_Classifier:

    def __init__(self, X: ndarray, y: ndarray, n_estimators: int, learning_rate: float):
        assert 0 < learning_rate <= 1  # 设定学习率阈值
        self.X = X
        self.y = y
        self.n_estimators = n_estimators
        self._w = np.full(shape=len(self.y), fill_value=1 / len(self.y))  # 初始化样本权重
        self.R = len(np.unique(self.y))  # 分类数目
        self.learning_rate = learning_rate

    def cart_tree(self, sample_weight):
        """
        :param sample_weight: 样本权重
        :return: 预测结果和分类误差率
        """
        cart_t = tree.DecisionTreeClassifier(max_depth=3)
        cart_t.fit(self.X, self.y, sample_weight=sample_weight)
        y_pred = cart_t.predict(self.X)
        e_k = ((y_pred != self.y) * sample_weight).sum()
        return y_pred, e_k, cart_t

    def alpha_k(self, sample_weight):
        """
        :param sample_weight: 样本权重
        :return: 第k个弱学习器的权重
        """
        _, e_k, _ = self.cart_tree(sample_weight=sample_weight)
        alpha_k = (0.5 * np.log((1 - e_k) / e_k) + np.log(self.R - 1)) * self.learning_rate
        return alpha_k

    def get_new_weight(self, alpha_k, y_pred, weight):
        """
        :param alpha_k: 分类器的权重
        :param y_pred: 该分类器的预测值
        :param weight: 样本权重
        :return: 更新后的样本权重
        """
        weight_new = weight * np.exp(alpha_k * (y_pred != self.y))
        weight_new /= weight_new.sum()
        return weight_new

    def fit(self):
        """
        :return: cart树模型和弱学习器的权重
        """
        sample_weight = self._w
        cart_list = []
        alpha_k_list = []
        for i in range(self.n_estimators):
            _, e_k, cart_t = self.cart_tree(sample_weight=sample_weight)
            alpha_k = self.alpha_k(sample_weight=sample_weight)
            weight_new = self.get_new_weight(alpha_k, _, sample_weight)
            print(f'第{i}个弱学习器构建完成， 该学习器的分类错误率为{e_k}， 权重为{alpha_k}', '\n')
            sample_weight = weight_new
            cart_list.append(cart_t)
            alpha_k_list.append(alpha_k)
        return cart_list, np.array(alpha_k_list)

    def predict(self, dataX):
        cart_list, alpha_k = self.fit()
        df_result = pd.DataFrame()
        df_result['alpha_k'] = alpha_k
        for h, data in enumerate(dataX):
            y_pred_list = []
            for i in range(self.n_estimators):
                y_pred = cart_list[i].predict([data])
                y_pred_list.append(y_pred[0])
            df_result[f'样本{h}'] = y_pred_list
        return df_result


if __name__ == '__main__':
    data = load_iris().data
    y = load_iris().target
    ada = Ada_CART_Classifier(data, y, 100, 0.8)
    print(ada.fit())
    data_X = np.array([[1, 2, 3, 4],
                       [3, 3, 4, 5]])
    print(ada.predict(data_X))
