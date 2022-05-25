import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', skiprows=1, names=['grade1', 'grade2', 'label'])

# 数据处理
label_list = []
for label in data['label']:
    label_list.append(re.findall(r'(\d)', label)[0])
data['label'] = label_list
print(data.head())

# 数据归一化和标准化
# 两者的区别在于归一化的鲁棒性较差，对如果出现异常值的数据会出现较大问题
# 而标准化通过方差和均值把数据转换为均值为0，标准差为1的范围内
# 1.归一化
norm_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
Norm_data = norm_scale.fit_transform(data[['grade1', 'grade2']])
Norm_data = pd.DataFrame.from_records(Norm_data, columns=['grade1', 'grade2'])
Norm_data['label'] = data['label']
print(Norm_data, '\n', '*'*30)
# 2.标准化
std_scale = preprocessing.StandardScaler()
Std_data = std_scale.fit_transform(data[['grade1', 'grade2']])
Std_data = pd.DataFrame.from_records(Std_data, columns=['grade1', 'grade2'])
print(Std_data)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(Norm_data[['grade1', 'grade2']], Norm_data['label'], random_state=1)
print(X_train, '\n', X_test)

# 用逻辑回归模型训练数据(交叉验证cv也包含在下面)
# scikit learn 包中默认采用的求解算法为lbfgs
logistic_model = LogisticRegression()
logistic_cv_model = LogisticRegressionCV(cv=10)

logistic_model.fit(X_train, Y_train)
norm_result = logistic_model.predict(X_test)
print(norm_result)
print('logistic_model score', logistic_model.score(X_test, Y_test))

logistic_cv_model.fit(X_train, Y_train)
cv_result = logistic_cv_model.predict(X_test)
print(cv_result)
print('logistic_cv_model score', logistic_cv_model.score(X_test, Y_test))


# 数据可视化
fig = plt.figure()
plt.style.use('ggplot')
ax = fig.add_subplot(projection='3d')
for grade1, grade2, label in zip(Norm_data['grade1'], Norm_data['grade2'], Norm_data['label']):
    print(grade1)
    ax.scatter(float(grade1), float(grade2), float(label), marker='o', c='r', s=80, edgecolor='k')
plt.show()

plt.scatter('grade1', 'grade2', data=Norm_data[Norm_data['label'] == '0'])
plt.scatter('grade1', 'grade2', data=Norm_data[Norm_data['label'] == '1'])
plt.show()

