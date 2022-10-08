import numpy as np


class LinearRegression:

    def __init__(self, X, y, iteration_num=1000, learning_rate=0.01, param_dict=None):
        if param_dict is None:
            self.param_dict = {'w': 0, 'b': 0}
        self.X = X
        self.y = y
        self.iteration_num = iteration_num
        self.learning_rate = learning_rate

    def error_loss(self, w: float, b: float):
        """
        根据当前w，b参数计算均方误差
        :return: 均方误差
        """
        totalError = 0
        for i in range(0, len(self.y)):
            totalError += (w * self.X[i] + b - self.y[i])**2

        return totalError/float(len(self.y))

    def gradient_descent(self, w: float, b: float):
        length = len(self.y)
        b_g, w_g = 0, 0
        for i in range(length):
            b_g += (2 / float(length)) * ((w * self.X[i] + b) - self.y[i])
            w_g += (2 / float(length)) * ((w * self.X[i] + b) - self.y[i]) * self.X[i]
        new_b = b - b_g*self.learning_rate
        new_w = w - w_g*self.learning_rate

        return new_w, new_b

    def fit(self):
        w, b = self.param_dict['w'], self.param_dict['b']

        for i in range(self.iteration_num):
            error = self.error_loss(w, b)
            w, b = self.gradient_descent(w, b)
            result_dict = {
                '损失': error,
                'w': w,
                'b': b
            }
            print(f'第{i}次迭代完成，本次的信息为{result_dict}')
        return w, b

    def predict(self):
        pass


if __name__ == '__main__':
    X = []
    y = []
    for i in range(100):
        X.append(np.random.uniform(-10., 10.0))
        eps = np.random.normal(0, 0.01)
        y.append(1.447*X[i] + 0.089 + eps)
    lr = LinearRegression(X, y)
    lr.fit()


