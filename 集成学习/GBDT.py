#!/skl/bin/python3.9
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate
import time


def get_data():
    file_path = 'D:/pycharmlearn/MachineLearning/ML06_集成学习/train_modified/train_modified.csv'  # 此处为文件路径
    org_data = pd.read_csv(file_path)
    # print(org_data.head(10))  查看数据前10行
    new_data = org_data.drop(columns='ID')
    X = new_data[[col for col in new_data.columns if col not in ['Disbursed']]]
    y = new_data['Disbursed']
    return X, y


def BestModel(X, y):
    param_dict = {
        'n_estimators': range(10, 81, 10),  # 弱学习器的个数
        'max_depth': range(3, 14, 2),  # 决策树的深度
        'min_samples_split': range(100, 801, 200),  # 在划分所需样本最小数目
        'min_samples_leaf': range(60, 101, 10),  # 叶子节点最小样本数
        'max_features': range(8, 20, 2),  # 最大特征数目
        'subsample': [0.6, 0.7, 0.8, 0.9, 0.85]  # 数据传入比例
    }

    # 网格化搜索参数
    g_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1), param_grid=param_dict,
                            scoring='roc_auc', cv=5)

    # 拟合模型
    g_search.fit(X=X, y=y)

    # 模型参数和得分
    return g_search.best_params_


def main():
    # 1.get_data()  得到数据集，并将数据集转换为模型可用的格式
    X, y = get_data()[0], get_data()[1]

    # 2.BestModel()  模型初始化以及参数选择,输出最优模型及其参数
    param_dict = BestModel(X, y)

    # 3.predict()   模型预测
    gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=param_dict['n_estimators'],
                                      max_depth=param_dict['max_depth'],
                                      min_samples_split=param_dict['min_samples_split'],
                                      min_samples_leaf=param_dict['min_samples_leaf'],
                                      max_features=param_dict['max_features'], subsample=param_dict['subsample'])

    print(param_dict)


if __name__ == '__main__':
    begin = time.time()

    main()

    over = time.time()

    print(f'本次训练花费的时间为{over - begin}')

