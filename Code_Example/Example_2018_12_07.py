
# coding: utf-8

# # datasets 模块常用数据集加载函数及其解释

# In[1]:


#          数据集加载函数                     数据集任务类型

#          load_boston                        回归

#          fetch_california_housing           回归

#          load_digits                        分类

#          load_breast_cancer                 分类、聚类

#          load_iris                          分类、聚类

#          load_wine                          分类


# # 加载 breast_cancer 数据集

# In[2]:


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer() # 将数据集赋值给 cancer 变量
print("breast_cancer 数据集的长度为：", len(cancer))
print("breast_cancer 数据集的类型为：", type(cancer))


# # sklearn 自带数据集内部信息获取

# In[3]:


cancer_data = cancer["data"]
print("breast_cancer 数据集的数据为：\n", cancer_data)


# In[4]:


cancer_target = cancer["target"] # 取出数据集的标签
print("breast_cancer 数据仅的标签为：\n", cancer_target)


# In[5]:


cancer_names = cancer["feature_names"] # 取出数据集的特征名
print("breast_cancer 数据集的特征名为：\n", cancer_names)


# In[6]:


cancer_desc = cancer["DESCR"] # 取出数据集的描述信息
print("breast_cancer 数据集的描述信息为：\n", cancer_desc)


# # train_test_split 常用参数及其说明

# In[7]:


# train_test_split 函数：

# sklearn.model_selection.train_test_split(*arrays, **options)

# *arrays          接收一个或多个数据集。代表需要划分的数据集。若为分类回归，则分别传入数据和标签；若为聚类，则传入数据。无默认

# test_size        接收 float、int 类型的数据或者 None。代表测试集的大小。如果传入的为 float 类型的数据，则需要限定在 0~1 之间，
#                  代表测试集在总数中的占比；如果传入的为 int 类型的数据，则表示测试集记录的绝对数目。该参数与 train_size 可以
#                  只传入一个。在 0.21 版本前，若 test_size 和 train_size 均为默认，则 test_size 为 25%

# train_size       接收 float、int 类型的数据或者 None。代表训练集的大小。该参数与 test_size 可以只传入一个

# random_state     接收 int。代表随机种子编号，相同随机种子编号产生相同的随机结果，不同的随机种子编号产生不同的随机结果。默认为 None

# shuffle          接收 boolean。代表是否进行有放回抽样。若该参数取值为 True，则 stratify 参数必须不能为空

# stratify         接收 array 或者 None。如果不为 None，则使用传入的标签进行分层抽样


# # 使用 train_test_split 划分数据集

# In[8]:


print("原始数据集数据的形状为：", cancer_data.shape)
print("原始数据集标签的形状为：", cancer_target.shape)


# In[10]:


from sklearn.model_selection import train_test_split

cancer_data_train, cancer_data_test, cancer_target_train, cancer_target_test = \
    train_test_split(cancer_data, cancer_target, test_size = 0.2, random_state = 42)
print("训练集数据的形状为：", cancer_data_train.shape)
print("训练集标签的形状为：", cancer_target_train.shape)
print("测试集数据的形状为：", cancer_data_test.shape)
print("测试集标签的形状为：", cancer_target_test.shape)


# # 转换器的 3 个方法及其说明

# In[11]:


# fit                fit 方法主要通过分析特征和目标值提取有价值的信息，这些信息可以是统计量，也可以是权值系数等

# transform          transform 方法主要用来对特征进行转换。从可利用信息的角度分为无信息转换和有信息转换。无信息转换是指不利用
#                    任何其他信息进行转换，比如指数和对数函数转换等。有信息转换根据是否利用目标值向量又可分为无监督转换和有监
#                    督转换。无监督转换指只利用特征的统计信息的转换，比如标准化和 PCA 降维等。有监督转换指既利用了特征信息又利
#                    用了目标值信息的转换，比如通过模型选择特征和 LDA 降维等

# fit_transform      fit_transform 方法就是先调用 fit 方法，然后调用 transform 方法


# # 离差标准化

# In[15]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler().fit(cancer_data_train) # 生成规则

# 将规则应用于训练集

cancer_trainScaler = Scaler.transform(cancer_data_train)

# 将规则应用于测试集

cancer_testScaler = Scaler.transform(cancer_data_test)

print("离差标准化前训练集数据的最小值为：", np.min(cancer_data_train))
print("离差标准化后训练集数据的最小值为：", np.min(cancer_trainScaler))
print("离差标准化前训练集数据的最大值为：", np.max(cancer_data_train))
print("离差标准化后训练集数据的最大值为：", np.max(cancer_trainScaler))
print("离差标准化前测试集数据的最小值为：", np.min(cancer_data_test))
print("离差标准化后测试集数据的最小值为：", np.min(cancer_testScaler))
print("离差标准化前测试集数据的最大值为：", np.max(cancer_data_test))
print("离差标准化后测试集数据的最大值为：", np.max(cancer_testScaler))


# # sklearn 部分预处理函数与其作用

# In[13]:


# StandardScaler           对特征进行标准差标准化

# Normalizer               对特征进行归一化

# Binarizer                对定量特征进行二值化处理

# OneHotEncoder            对定性特征进行独热编码处理

# FunctionTransformer      对特征进行自定义函数变换


# # 对 breast_cancer 数据集 PCA 降维

# In[16]:


from sklearn.decomposition import PCA

pca_model = PCA(n_components = 10).fit(cancer_trainScaler) # 生成规则

# 将规则应用于训练集

cancer_trainPca = pca_model.transform(cancer_trainScaler)

# 将规则应用于测试集

cancer_testPca = pca_model.transform(cancer_testScaler)

print("PCA 降维前训练集数据的形状为：", cancer_trainScaler.shape)
print("PCA 降维后训练集数据的形状为：", cancer_trainPca.shape)
print("PCA 降维前测试集数据的形状为：", cancer_testScaler.shape)
print("PCA 降维后测试集数据的形状为：", cancer_testPca.shape)


# # PCA 降维算法函数常用参数及其作用

# In[17]:


# n_components          接收 None、int、float 或 mle。未指定时，代表所有特征均会被保留下来；如果为 int，则表示将原始数据降低到 n 个维度；
#                       如果为 float，则 PCA 根据样本特征方差来决定降维后的维度数；赋值为 "mle"，PCA 会用 MLE 算法根据特征的方差分布情况
#                       自动选择一定数量的主成分特征来降维。默认为 None

# copy                  接收 boolean。代表是否在运行算法时降原始数据复制一份，如果为 True，则运行后，原始数据的值不会有任何改变；如果为
#                       False，则运行 PCA 算法后，原始数据的值会发生改变。默认为 True

# whiten                接收 boolean。表示白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为 1。默认为 False

# svd_solver            接收 auto、full、arpack、randomized。代表使用的 SVD 算法。randomized 一般适用于数据量大，数据维度多，同时主成分数
#                       目比例又较低的 PCA 降维，它使用了一些加快 SVD 的随机算法。full 是使用 SciPy 库实现的传统 SVD 算法。arpack 和
#                       randomized 的适用场景类似，区别是，randomized 使用的是 sklearn 自己的 SVD 实现，而 arpack 直接使用了 SciPy 库的
#                       sparse SVD 实现。auto 则代表 PCA 类会自动在上述 3 种算法中去权衡，选择一个合适的 SVD 算法来降维。默认为 auto


# # 获取 sklearn 自带的 boston 数据集

# In[19]:


from sklearn.datasets import load_boston

boston = load_boston()
boston_data = boston["data"]
boston_target = boston["target"]
boston_names = boston["feature_names"]

print("boston 数据集数据的形状为：", boston_data.shape)
print("boston 数据集标签的形状为：", boston_target.shape)
print("boston 数据集特征名的形状为：", boston_names.shape)


# # 使用 train_test_split 划分 boston 数据集

# In[20]:


from sklearn.model_selection import train_test_split

boston_data_train, boston_data_test, boston_target_train, boston_target_test =     train_test_split(boston_data, boston_target, test_size = 0.2, random_state = 42)

print("训练集数据的形状为：", boston_data_train.shape)
print("训练集标签的形状为：", boston_target_train.shape)
print("测试集数据的形状为：", boston_data_test.shape)
print("测试集标签的形状为：", boston_target_test.shape)


# # 使用 stdScale.transform 进行数据预处理

# In[22]:


from sklearn.preprocessing import StandardScaler

stdScale = StandardScaler().fit(boston_data_train) # 生成规则

# 将规则应用于训练集

boston_trainScaler = stdScale.transform(boston_data_train)

# 将规则应用于测试集

boston_testScaler = stdScale.transform(boston_data_test)

print("标准差标准化后训练集数据的方差为：", np.var(boston_trainScaler))
print("标准差标准化后训练集数据的均值为：", np.mean(boston_trainScaler))
print("标准差标准化后测试集数据的方差为：", np.var(boston_testScaler))
print("标准差标准化后测试集数据的均值为：", np.mean(boston_testScaler))


# # 使用 pca.transform 进行 PCA 降维

# In[23]:


from sklearn.decomposition import PCA

# 生成规则

pca = PCA(n_components = 5).fit(boston_trainScaler)

# 将规则应用于训练集

boston_trainPca = pca.transform(boston_trainScaler)

# 将规则应用于测试集

boston_testPca = pca.transform(boston_testScaler)

print("降维后 boston 数据集数据测试集的形状为：", boston_trainPca.shape)
print("降维后 boston 数据集数据训练集的形状为：", boston_testPca.shape)

