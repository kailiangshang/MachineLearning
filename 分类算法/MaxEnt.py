"""
_*_ coding: utf-8 _*_
author: skl
abstract:采用IIS算法构建最大熵模型，并应用到鸢尾花数据集上。
"""
import math
from collections import defaultdict
import numpy as np


class MaxEnt:
    def __init__(self, epsilon=1e-3, max_step=100):
        self.epsilon = epsilon
        self.max_step = max_step
        self.w = None  # 特征函数权重
        self.labels = None  # 标签/分类
        self.fea_list = []  # 特征函数
        self.Px = defaultdict(lambda: 0)  # 经验边缘分布概率
        self.Pxy = defaultdict(lambda: 0)  # 经验联合分布概率，由于特征函数的取值为0/1，所以经验联合分布概率也为经验联合分布期望
        self.exp_fea = defaultdict(lambda: 0)  # 每个特征在数据集上的期望
        self.data_list = []  # 样本集合，
        self.N = None  # 样本总数
        self.M = None  # 某个训练样本包含的特征总数
        self.n_fea = None  # 样本特征函数的总数

    def init_param(self, X, y):
        """
        根据传入的样本集，初始化模型参数。
        无返回值。
        """
        self.N = X.shape[0]
        self.labels = np.unique(y)

        self.n_fea = len(self.fea_list)
        return

    def _fea_func(self, X, y):
        """
        特征函数
        """
        self.M = X.shape[1]
        for xx, yy in zip(X, y):
            xx = tuple(xx)
            self.Px[xx] += 1 / self.N  # 计算X的经验边缘分布
            self.Pxy += 1 / self.N  # 计算X，y的经验联合分布
            for dim, val in enumerate(xx):
                key = (dim, val, y)
                if key not in self.fea_list:
                    self.fea_list.append(key)

    def _exp_fea(self, X, y):
        """
        计算特征的经验期望值
        """
        for xx, yy in zip(X, y):
            for dim, val in enumerate(xx):
                fea = (dim, val, y)
                self.exp_fea[fea] += self.Pxy[(tuple(xx), y)]

    def _Py_x(self, ):
        """
        计算当前w下的条件分布概率
        """

    def _est_fea(self):
        """
        计算每个特征的估计期望值
        """

    def IIS(self):
        """
        IIS算法
        """

    def fit(self, X, y):
        """
        训练模型
        """

    def predict(self, X):
        """
        输入特征向量X， 返回所有可能的条件概率
        """
