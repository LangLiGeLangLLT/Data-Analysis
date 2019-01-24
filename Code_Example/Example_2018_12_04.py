
# coding: utf-8

# # get_dummies 函数的参数及说明

# In[1]:


# get_dummies 函数：

# pandas.get_dummies(data, prefix = None, prefix_sep = "_", dummy_na = False, columns = None, sparse = False, drop_first = False)

# data           接收 array、DataFrame 或者 Series。表示需要哑变量处理的数据。无默认

# prefix         接收 string、string 的列表或者 string 的 dict。表示哑变量化后列名的前缀。默认为 None

# prefix_sep     接收 string。表示前缀的连接符。默认为 "_"

# dummy_na       接收 boolean。表示是否为 NaN 值添加一列。默认为 False

# columns        接收类似 list 的数据。表示 DataFrame 中需要编码的列名。默认为 None，表示对所有 object 和 category 类型进行编码

# sparse         接收 boolean。表示虚拟列是否是稀疏的。默认为 False

# drop_first     接收 boolean。表示是否通过从 k 个分类级别中删除第一级来获得 k-1 个分类级别。默认为 False


# # 哑变量处理示例

# In[2]:


import pandas as pd
import numpy as np

detail = pd.read_csv("../data/detail.csv", encoding = "gbk")
data = detail.loc[0:5, "dishes_name"] # 抽取部分数据做演示
print("哑变量处理前的数据为：\n", data)
print("哑变量处理后的数据为：\n", pd.get_dummies(data))


# # cut 函数的常用参数及其说明

# In[3]:


# cut 函数：

# pandas.cut(x, bins, right = True, labels = None, retbins = False, precision = 3,include_lowest = False)

# x                     接收 array 或 Series。代表需要进行离散化处理的数据。无默认

# bins                  接收 int、list、array 和 tuple。若为 int，则代表离散化后的类别数目；若为序列类型的数据，则表示
#                       进行切分的区间，每两个数的间隔为一个区间。无默认

# right                 接收 boolean。代表右侧是否为闭区间。默认为 True

# labels                接收 list、array。代表离散化后各个类别的名称。默认为空

# retbins               接收 boolean。代表是否返回区间标签。默认为 False

# precision             接收 int。显示标签的精度。默认为 3


# # 等宽法离散化示例

# In[4]:


price = pd.cut(detail["amounts"], 5)
print("离散化后 5 条记录售价分布为：\n", price.value_counts())


# # 等频法离散化示例

# In[5]:


# 自定义等频法离散化函数

def SameRateCut(data, k):
    w = data.quantile(np.arange(0, 1 + 1.0 / k, 1.0 / k))
    data = pd.cut(data, w)
    return data

# 对菜品售价进行等频法离散化

result = SameRateCut(detail["amounts"], 5).value_counts()
print("菜品数据等频法离散化各个类别数目分布状况为：\n", result)


# # 基于聚类分析的离散化

# In[21]:


# 自定义数据 K-Means 聚类离散化函数

def KmeanCut(data, k):
    from sklearn.cluster import KMeans # 引入 K-Means
    # 建立模型，n_jobs 是并行数
    kmodel = KMeans(n_clusters = k, n_jobs = 4)
    kmodel.fit(data.reshape((len(data), 1))) # 训练模型
    # 输出聚类中心并排序
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
    w = pd.rolling_mean(c, 2).iloc[1:] # 相邻两项求中点，作为边界点
    w = [0] + list(w[0]) +[data.max()] # 把首末边界点加上
    data = pd.cut(data, w)
    return data

# 菜品售价等频法离散化
result = KmeanCut(detail["amounts"], 5).value_counts()
print("菜品售价聚类离散化后各个类别数目分布状况为：\n", result)


# # 菜品 dishes_name 哑变量处理

# In[7]:


data = detail.loc[:, "dishes_name"]
print("哑变量处理前的数据为：\n", data.iloc[:5])
print("哑变量处理后的数据为：\n", pd.get_dummies(data).iloc[:5, :5])


# # 菜品售价等频法离散化

# In[9]:


# 自定义等频法离散化函数

def SameRateCut(data, k):
    w = data.quantile(np.arange(0, 1 + 1.0 / k, 1.0 / k))
    data = pd.cut(data, w)
    return data

# 菜品售价等频法离散化

result = SameRateCut(detail["amounts"], 5).value_counts()
print("菜品数据等频法离散化后各个类别数目分布状况为：\n", result)

