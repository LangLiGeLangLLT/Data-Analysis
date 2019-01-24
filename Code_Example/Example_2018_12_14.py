
# coding: utf-8

# # 聚类算法类别

# In[1]:


#         算法类别                        包括的主要算法

#     划分（分裂）方法             K-Means 算法（K-平均）、K-MEDOIDS 算法（K-中心点）和 CLARANS 算法（基于选择的算法）

#     层次分析方法                 BIRCH 算法（平衡迭代规约和聚类）、CURE 算法（代表点聚类）和 CHAMELEON 算法（动态模型）

#     基于密度的方法               DBSCAN 算法（基于高密度连接区域）、DENCLUE 算法（密度分布函数）和 OPTICS 算法（对象排序识别）

#     基于网格的方法               STING 算法（统计信息网络）、CLIOUE 算法（聚类高维空间）和 WAVE-CLUSTER 算法（小波变换）


# # cluster 提供的聚类算法及其适用范围

# In[2]:


#         函数名称                        参数                                适用范围                                 距离度量

#         K-MEANS                         簇数                可用于样本数目很大、聚类数目中等的场景                 点之间的距离

#     Spectral clustering                 簇数                可用于样本数目中等、聚类数目较小的场景                    图距离

#  Ward hierarchical clustering           簇数                可用于样本数目中等、聚类数目较大的场景                 点之间的距离

#   Agglomerative clustering     簇数、链接类型、距离         可用于样本数目较大、聚类数目较大的场景            任意成对点线图间的距离

#         DBSCAN                半径大小、最低成员数目        可用于样本数目很大、聚类数目中等的场景              最近的点之间的距离

#         Birch              分支因子、阈值、可选全局集群     可用于样本数目很大、聚类数目较大的场景               点之间的欧式距离


# # 估计器两个方法的说明

# In[3]:


#         方法名称                                       说明

#           fit              fit 方法主要用于训练算法。该方法可接收用于有监督学习的训练集及其标签两个参数，也可以接收用于无监督学习的数据

#         predict            predict 用于预测有监督学习的测试集标签，亦可以用于划分传入数据的类别


# # 使用 sklearn 估计器构建 K-Means 聚类模型

# In[4]:


from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

iris = load_iris()
iris_data = iris["data"] # 提取数据集中的特征
iris_target = iris["target"] # 提取数据集中的标签
iris_names = iris["feature_names"] # 提取特征名

scale = MinMaxScaler().fit(iris_data) # 训练规则
iris_dataScale = scale.transform(iris_data) # 应用规则
kmeans = KMeans(n_clusters = 3, random_state = 123).fit(iris_dataScale) # 构建并训练模型
print("构建的 K-Means 模型为：\n", kmeans)


# In[5]:


result = kmeans.predict([[1.5, 1.5, 1.5, 1.5]])
print("花瓣花萼长度宽度全为 1.5 的鸢尾花预测类别为：", result[0])


# # 聚类结果可视化

# In[8]:


import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 使用 TSNE 进行数据降维，降成两维

tsne = TSNE(n_components = 2, init = "random", random_state = 177).fit(iris_data)
df = pd.DataFrame(tsne.embedding_) # 将原始数据转换为 DataFrame
df["labels"] = kmeans.labels_ # 将聚类结果存储进 df 数据表

# 提取不同标签的数据

df1 = df[df["labels"] == 0]
df2 = df[df["labels"] == 1]
df3 = df[df["labels"] == 2]

# 绘制图形

fig = plt.figure(figsize = (9, 6)) # 设定空白画布，并制定大小

# 用不同的颜色表示不同数据

plt.plot(df1[0], df1[1], "bo", df2[0], df2[1], "r*", df3[0], df3[1], "gD")
plt.savefig("../tmp/聚类结果.png")
plt.show() # 显示图片


# # metrics 模块提供的聚类模型评价指标

# In[9]:


#         方法名称                      真实值                  最佳值                   sklearn 函数

#    ARI 评价法（兰德系数）             需要                     1.0                  adjusted_rand_score

#    AMI 评价法（互信息）               需要                     1.0                  adjusted_mutual_info_score

#    V-measure评分                      需要                     1.0                  completeness_score

#    FMI 评价法                         需要                     1.0                  fowlkes_mallows_score

#    轮廓系数评价法                     不需要              畸变程度最大              silhouette_score

#    Calinski-Harabasz 指数评价法       不需要                相较最大                calinski_harabaz_score


# # 使用 FMI 评价法评价 K-Means 聚类模型

# In[11]:


from sklearn.metrics import fowlkes_mallows_score

for i in range(2, 7):
    # 构建并训练模型
    kmeans = KMeans(n_clusters = i, random_state = 123).fit(iris_data)
    score = fowlkes_mallows_score(iris_target, kmeans.labels_)
    print("iris 数据聚 %d 类 FMI 评价分值为：%f" %(i, score))


# # 使用轮廓系数评价法评价 K-Means 聚类模型

# In[12]:


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouettteScore = []

for i in range(2, 15):
    # 构建并训练模型
    kmeans = KMeans(n_clusters = i, random_state = 123).fit(iris_data)
    score = silhouette_score(iris_data, kmeans.labels_)
    silhouettteScore.append(score)

plt.figure(figsize = (10, 6))
plt.plot(range(2, 15), silhouettteScore, linewidth = 1.5, linestyle = "-")
plt.show()


# # 使用 Calinski-Harabasz 指数评价 K-Means 聚类模型

# In[14]:


from sklearn.metrics import calinski_harabaz_score

for i in range(2, 7):
    # 构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state = 123).fit(iris_data)
    score = calinski_harabaz_score(iris_data, kmeans.labels_)
    print("iris 数据聚 %d 类 calinski_harabaz 指数为：%f" %(i, score))


# # 对 Seeds 构建 K-Means 聚类模型

# In[16]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

seeds = pd.read_csv("../data/seeds_dataset.txt", sep = "\t")
print("数据集形状为：", seeds.shape)

# 处理数据

seeds_data = seeds.iloc[:, :7].values
seeds_target = seeds.iloc[:, 7].values
seeds_names = seeds.columns[:7]
stdScale = StandardScaler().fit(seeds_data)
seeds_dataScale = stdScale.transform(seeds_data)

# 构建并训练模型

kmeans = KMeans(n_clusters = 3, random_state = 42).fit(seeds_data)
print("构建 K-Means 模型为：\n", kmeans)


# # 评价使用 Seeds 数据集构建的 K-Means 聚类模型

# In[17]:


from sklearn.metrics import calinski_harabaz_score

for i in range(2, 7):
    # 构建并训练模型
    kmeans = KMeans(n_clusters = i, random_state = 123).fit(seeds_data)
    score = calinski_harabaz_score(seeds_data, kmeans.labels_)
    print("seeds 数据聚 %d 类 calinski_harabaz 指数为：%f" %(i, score))


# # sklearn 库的常用分类算法

# In[18]:


#            模块名称                 函数名称                  算法名称

#         linear_model           LogisticRegression           逻辑斯蒂回归

#             svm                       SVC                    支持向量机

#           neighbors            KNeighborsClassifier          K 最近邻分类

#          native_bayes              GaussianNB               高斯朴素贝叶斯

#            tree              DecisionTreeClassifier           分类决策树

#           ensemble           RandomForestClassifier          随机森林分类

#           ensemble          GradientBoostingClassifier      梯度提升分类树


# # 使用 sklearn 估计器构建 SVM 模型

# In[19]:


# 加载所需的函数

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
cancer_data = cancer["data"]
cancer_target = cancer["target"]
cancer_names = cancer["feature_names"]

# 将数据划分为训练集、测试集

cancer_data_train, cancer_data_test, cancer_target_train, cancer_target_test = \
    train_test_split(cancer_data, cancer_target, test_size = 0.2, random_state = 22)

# 数据标准化

stdScaler = StandardScaler().fit(cancer_data_train)
cancer_trainStd = stdScaler.transform(cancer_data_train)
cancer_testStd = stdScaler.transform(cancer_data_test)

# 建立 SVM 模型

svm = SVC().fit(cancer_trainStd, cancer_target_train)
print("建立的 SVM 模型为：\n", svm)


# In[20]:


# 预测训练集结果

cancer_target_pred = svm.predict(cancer_testStd)
print("预测 20 个结果为：\n", cancer_target_pred[:20])


# # 分类结果的混淆矩阵与准确率

# In[21]:


# 求出预测和真实一样的数目

true = np.sum(cancer_target_pred == cancer_target_test)
print("预测对的结果数目为：", true)
print("预测错的结果数目为：", cancer_target_test.shape[0] - true)
print("预测结果准确率为：", true / cancer_target_test.shape[0])


# # 分类模型评价方法

# In[22]:


#            方法名称                  最佳值                   sklearn函数

#        Precision(精确率)              1.0                metrics.precision_score

#        Recall（召回率）               1.0                metrics.recall_score

#            F1 值                      1.0                metrics.f1_score

#         Cohen's Kappa 系数            1.0                metrics.cohen_kappa_score

#           ROC 曲线                 最靠近 y 轴           metrics.roc_curve


# # 分类模型常用评价方法

# In[24]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

print("使用 SVM 预测 breast_cancer 数据的准确率为：", accuracy_score(cancer_target_test, cancer_target_pred))
print("使用 SVM 预测 breast_cancer 数据的精确率为：", precision_score(cancer_target_test, cancer_target_pred))
print("使用 SVM 预测 breast_cancer 数据的召回率为：", recall_score(cancer_target_test, cancer_target_pred))
print("使用 SVM 预测 breast_cancer 数据的 F1 值为：", f1_score(cancer_target_test, cancer_target_pred))
print("使用 SVM 预测 breast_cancer 数据的 Cohen's Kappa 系数为：", cohen_kappa_score(cancer_target_test, cancer_target_pred))


# # 分类模型评价报告

# In[25]:


from sklearn.metrics import classification_report

print("使用 SVM 预测 iris 数据的分类报告为：\n", classification_report(cancer_target_test, cancer_target_pred))


# # 绘制 ROC 曲线

# In[26]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# 求出 ROC 曲线的 x 轴和 y 轴

fpr, tpr, thresholds = roc_curve(cancer_target_test, cancer_target_pred)

plt.figure(figsize = (10, 6))
plt.xlim(0, 1) # 设定 x 轴的范围
plt.ylim(0.0, 1.1) # 设定 y 轴的范围
plt.xlabel("False Postive Rate")
plt.ylabel("True Postive Rate")
plt.plot(fpr, tpr, linewidth = 2, linestyle = "-", color = "red")
plt.show()


# # 鲍鱼年龄预测

# In[27]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

abalone = pd.read_csv("../data/abalone.data", sep = ",")

# 将数据和标签拆开

abalone_data = abalone.iloc[:, :8]
abalone_target = abalone.iloc[:, 8]

# 连续型特征离散化

sex = pd.get_dummies(abalone_data[["sex"]])
abalone_data = pd.concat([abalone_data, sex], axis = 1)
abalone_data.drop("sex", axis = 1, inplace = True)

# 划分训练集、测试集

abalone_train, abalone_test, abalone_target_train, abalone_target_test = \
    train_test_split(abalone_data, abalone_target, train_size = 0.8, random_state = 42)

# 标准化

stdScaler = StandardScaler().fit(abalone_train)
abalone_std_train = stdScaler.transform(abalone_train)
abalone_std_test = stdScaler.transform(abalone_test)

# 建模

svm_abalone = SVC().fit(abalone_std_train, abalone_target_train)
print("建立的 SVM 模型为：\n", svm_abalone)


# # 评价构建的 SVM 分类模型

# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

stdScale = StandardScaler().fit(abalone_test) # 生成规则
abalone_testScaler = stdScale.transform(abalone_test) # 将规则应用于测试集

pca_model= PCA(n_components = 10).fit(abalone_testScaler) # 生成规则
abalone_testPca = pca_model.transform(abalone_testScaler) # 将规则应用于测试集

abalone_target_pred = svm_abalone.predict(abalone_testPca)
print("abalone 数据集的 SVM 分类报告为：\n", classification_report(abalone_target_test, abalone_target_pred))


# # 常用的回归模型

# In[1]:


#  回归模型名称                    适用条件                                                  算法描述

#   线性回归             因变量与自变量是线性关系            对一个或多个自变量和因变量之间的线性关系进行建模，可用最小二乘估计法求解模型系数

#  非线性回归            因变量与自变量之间不都是线性关系    对一个或多个自变量和因变量之间的非线性关系进行建模。如果非线性关系可以通过简单的
#                                                            函数变换转化成线性关系，则可用线性回归的思想求解；如果不能转化，可用非线性最小二
#                                                            乘估计法求解

# Logistic 回归          因变量一般有 1 和 0（是与否）两     是广义线性回归模型的特例，利用 Logistics 函数将因变量的取值范围控制在 0~1，表示取
#                        种取值                              值为 1 的概率

#   岭回归               参与建模的自变量之间具有多重共线性  是一种改进最小二乘估计法的方法

#  主成分回归            参与建模的自变量之间具有多重共线性  主成分回归是根据主成分分析的思想提出来的，是对最小二乘估计法的一种改进，它是参数估
#                                                            计。可以消除自变量之间的多重共线性


# # sklearn 库内部的常用回归算法

# In[2]:


#           模块名称                  函数名称                 算法名称

#        linear_model              LinearRegression            线性回归

#            svm                        SVR                    支持向量回归

#         neighbors               KNeighborsRegressor          最近邻回归

#            tree               DecisionTreeRegressor          回归决策树

#          ensemble             RandomForestRegressor          随机森林回归

#          ensemble            GradientBoostingRegressor       梯度提升回归树


# # 使用 sklearn 估计器构建线性回归模型

# In[5]:


# 加载所需函数

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 加载 boston 数据

boston = load_boston()
X = boston["data"]
y = boston["target"]
names = boston["feature_names"]

# 将数据划分为训练集、测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 125)

# 建立线性回归模型

clf = LinearRegression().fit(X_train, y_train)
print("建立的 Linear Regression 模型为：\n", clf)


# In[6]:


# 预测训练集结果

y_pred = clf.predict(X_test)
print("预测前 20 个结果为：\n", y_pred[:20])


# # 回归结果可视化

# In[8]:


import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = "SimHei"
fig = plt.figure(figsize = (10, 6)) # 设定空白画布，并制定大小
plt.plot(range(y_test.shape[0]), y_test, color = "blue", linewidth = 1.5, linestyle = "-")
plt.xlim((0, 102))
plt.ylim((0, 55))
plt.legend(["真实值", "预测值"])
plt.savefig("../tmp/聚会归类结果.png")
plt.show() # 显示图片


# # 回归模型评价指标

# In[9]:


#         方法名称             最优值                 sklearn 函数

#       平均绝对误差            0.0              metrics.mean_absolute_error

#       均方误差                0.0              metrics.mean_squared_error

#       中值绝对误差            0.0              metrics.median_absolute_error

#       可解释方差值            1.0              metrics.explained_variance_score

#          R ^ 2 值                1.0              metrics.r2_score


# # 回归模型评价

# In[10]:


from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

print("Boston 数据线性回归模型的平均绝对误差为：", mean_absolute_error(y_test, y_pred))
print("Boston 数据线性回归模型的均方误差为：", mean_squared_error(y_test, y_pred))
print("Boston 数据线性回归模型的中值绝对误差为：", median_absolute_error(y_test, y_pred))
print("Boston 数据线性回归模型的可解释方差值为：", explained_variance_score(y_test, y_pred))
print("Boston 数据线性回归模型的 R ^ 2 值为：", r2_score(y_test, y_pred))


# # 使用 sklearn 估计器构建梯度提升回归模型

# In[11]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

house = pd.read_csv("../data/cal_housing.data", sep = ",")
house_data = house.iloc[:, :-1]
house_target = house.iloc[:, -1]
house_names = ["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", "medianIncome"]

house_train, house_test, house_target_train, house_target_test = \
    train_test_split(house_data, house_target, test_size = 0.2, random_state = 42)
GBR_house = GradientBoostingRegressor().fit(house_train, house_target_train)
print("建立的梯度提升回归模型为：\n", GBR_house)


# # 评价构建梯度提升回归模型

# In[12]:


house_target_pred = GBR_house.predict(house_test)

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

print("california_housing 数据梯度提升回归树模型的平均绝对误差为：", mean_absolute_error(house_target_test, house_target_pred))
print("california_housing 数据梯度提升回归树模型的均方误差为：", mean_squared_error(house_target_test, house_target_pred))
print("california_housing 数据梯度提升回归树模型的中值绝对误差为：", median_absolute_error(house_target_test, house_target_pred))
print("california_housing 数据梯度提升回归树模型的可解释方差值为：", explained_variance_score(house_target_test, house_target_pred))
print("california_housing 数据梯度提升回归树模型的 R ^ 2 值为：", r2_score(house_target_test, house_target_pred))

