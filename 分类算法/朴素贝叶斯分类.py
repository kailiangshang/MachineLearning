"""
_*_ coding:utf-8 _*_
author: skl
creat T: 2022-5-10
complete T : 2022-5-10
abstract: 通过sklearn包中的自带鸢尾花数据集实现不同贝叶斯回归模型，并对比得分。
        并对比对于不同特征使用不同模型下的得分。（）
"""
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler

# 导入数据集,并对数据特征进行标准化
data = datasets.load_iris()
X = StandardScaler().fit_transform(X=data.data)
X = pd.DataFrame.from_records(X, columns=data.feature_names)
y = data.target
print(X)
# 划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 模型拟合（高斯，多项式，伯努利）
# 高斯
# 查看具体概率可以通过调用predict_proba
gauss_clf = GaussianNB()
gauss_clf.fit(X_train, y_train)
gauss_score = gauss_clf.score(X_test, y_test)

# 伯努利
bernoulli_model = BernoulliNB()
bernoulli_model.fit(X_train, y_train)
bernoulli_score = bernoulli_model.score(X_test, y_test)
print(bernoulli_score)

# 多项朴素贝叶斯
# (以下代码报错，主要原因是多项式贝叶斯通常需要整数特征计数，多用于文本分类)
multi_model = MultinomialNB()
multi_model.fit(X_train, y_train)
multi_score = multi_model.score(X_test, y_test)
print(multi_score)








