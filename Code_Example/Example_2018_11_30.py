
# coding: utf-8

# # 利用 list 去重

# In[1]:


import pandas as pd

detail = pd.read_csv("../data/detail.csv", index_col = 0, encoding = "gbk")

# 方法一
# 定义去重函数

def delRep(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2

# 去重

dishes = list(detail["dishes_name"]) # 将 dishes_name 从数据框中提取出来
print("去重前菜品总数为：", len(dishes))
dish = delRep(dishes) # 使用自定义的去重函数去重
print("方法一去重后菜品总数为：", len(dish))


# # 利用 set 的特性去重

# In[2]:


# 方法二

print("去重前菜品总数为：", len(dishes))
dish_set = set(dishes) # 利用 set 的特性去重
print("方法二去重后菜品总数为：", len(dish_set))


# # 使用 drop_duplicates 方法对菜品名称去重

# In[3]:


# 对 dishes_name 去重

dishes_name = detail["dishes_name"].drop_duplicates()
print("drop_duplicates 方法去重之后菜品总数为：", len(dishes_name))


# # 使用 drop_duplicates 方法对多列去重

# In[4]:


print("去重之前订单详情表的形状为：", detail.shape)
shapeDet = detail.drop_duplicates(subset = ["order_id", "emp_id"]).shape
print("依照订单编号，会员编号去重之后订单详情表大小为：", shapeDet)


# # 求出 counts 和 amounts 两列数据的 kendall 法相似度矩阵

# In[5]:


# 求出销量和售价的相似度

corrDet = detail[["counts", "amounts"]].corr(method = "kendall")
print("销量和售价的 kendall 法相似度矩阵为：\n", corrDet)


# # 求出 dishes_name、counts 和 amounts 这 3 个特征的 pearson 法相似度矩阵

# In[6]:


corrDet1 = detail[["dishes_name", "counts", "amounts"]].corr(method = "pearson")
print("菜品名称、销量和售价的 pearson 法相似度矩阵为：\n", corrDet1)


# # 使用 DataFrame.equals 方法去重

# In[7]:


# 定义求取特征是否完全相同的矩阵的函数

def FeatureEquals(df):
    dfEquals = pd.DataFrame([], columns = df.columns, index = df.columns)
    for i in df.columns:
        for j in df.columns:
            dfEquals.loc[i, j] = df.loc[:, i].equals(df.loc[:, j])
    return dfEquals

# 应用上述函数

detEquals = FeatureEquals(detail)
print("detail 的特征相等矩阵的前 5 行 5 列为：\n", detEquals.iloc[:5, :5])


# # 通过遍历的方式进行数据筛选

# In[8]:


# 遍历所有数据

lenDet = detEquals.shape[0]
dupCol = []
for k in range(lenDet):
    for l in range(k + 1, lenDet):
        if detEquals.iloc[k, l] & (detEquals.columns[l] not in dupCol):
            dupCol.append(detEquals.columns[l])
            
# 进行去重操作

print("需要删除的列为：", dupCol)
detail.drop(dupCol, axis = 1, inplace = True)
print("删除多余列后 detail 的特征数目为：", detail.shape[1])


# # isnull 和 notnull 用法

# In[9]:


print("detail 每个特征缺失的数目为：\n", detail.isnull().sum())
print("detail 每个特征非缺失的数目为：\n", detail.notnull().sum())


# # dropna 主要参数及其说明

# In[34]:


# dropna 函数：

# pandas.DataFrame.dropna(self, axis = 0, how = "any", thresh = None, subset = None, inplace = False)

# axis                接收 0 或 1。表示轴向，0 为删除观测记录（行），1 为删除特征（列）。默认为 0

# how                 接收待定 string。表示删除的形式。any 表示只要有缺失值存在就执行删除操作；all 表示
#                     当且仅当全部为缺失值时才执行删除操作。默认为 any

# subset              接收 array。表示进行去重的列/行。默认为 None，表示所有列/行

# inplace             接收 boolean。表示是否在原表上进行操作。默认为 False


# # 使用 dropna 方法删除缺失值

# In[10]:


print("去除缺失的列前 detail 的形状为：", detail.shape)
print("去除缺失的列后 detail 的形状为：", detail.dropna(axis = 1, how = "any").shape)


# # fillna 主要参数及其说明

# In[36]:


# fillna 函数：

# pandas.DataFrame.fillna(value = None, method = None, axis = None, inplace = False, limit = None)

# value         接收待定 string
#               backfill 或 bfill 表示使用下一个非缺失值来填补缺失值
#               pad 或 ffill 表示使用上一个非缺失值来填补缺失值。默认为 None

# axis          接收 0 或 1。表示轴向。默认为 1

# inplace       接收 boolean。表示是否在原表上进行操作。默认为 False

# limit         接收 int。表示填补缺失值个数上限，超过则不进行填补。默认为 None


# # 使用 fillna 方法替换缺失值

# In[11]:


detail = detail.fillna(-99)
print("detail 每个特征缺失的数目为：\n", detail.isnull().sum())


# # Scipy interpolate 模块插值

# In[12]:


# 线性插值

import numpy as np
from scipy.interpolate import interp1d

x = np.array([1, 2, 3, 4, 5, 8, 9, 10]) # 创建自变量 x
y1 = np.array([2, 8, 18, 32, 50, 128, 162, 200]) # 创建因变量 y1
y2 = np.array([3, 5, 7, 9, 11, 17, 19, 21]) #  创建因变量 y2
LinearInsValue1 = interp1d(x, y1, kind = "linear") # 线性插值拟合 x、y1
LinearInsValue2 = interp1d(x, y2, kind = "linear") # 线性插值拟合 x、y2
print("当 x 为 6、7 时，使用线性插值 y1 为：", LinearInsValue1([6, 7]))
print("当 x 为 6、7 时，使用线性插值 y2 为：", LinearInsValue2([6, 7]))


# In[13]:


# 拉格朗日插值

from scipy.interpolate import lagrange

LargeInsValue1 = lagrange(x, y1) # 拉格朗日插值拟合 x、y1
LargeInsValue2 = lagrange(x, y2) # 拉格朗日插值拟合 x、y2
print("当 x 为 6，7 时，使用拉格朗日插值 y1 为：", LargeInsValue1([6, 7]))
print("当 x 为 6，7 时，使用拉格朗日插值 y2 为：", LargeInsValue2([6, 7]))


# In[14]:


# 样条插值

from scipy.interpolate import spline

# 样条插值拟合 x、y1

SplineInsValue1 = spline(x, y1, xnew = np.array([6, 7]))

# 样条插值拟合 x、y2

SplineInsValue2 = spline(x, y2, xnew = np.array([6, 7]))
print("当 x 为 6，7 时，使用样条插值 y1 为：", SplineInsValue1)
print("当 x 为 6，7 时，使用样条插值 y2 为：", SplineInsValue2)


# # 使用 3σ 原则识别异常值

# In[15]:


# 定义 3σ 原则来识别异常值函数

def outRange(Ser1):
    boolInd = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.var() < Ser1)
    index = np.arange(Ser1.shape[0])[boolInd]
    outrange = Ser1.iloc[index]
    return outrange
outlier = outRange(detail["counts"])
print("使用 3σ 原则判断异常值个数为：", outlier.shape[0])
print("异常值的最大值为：", outlier.max())
print("异常值的最小值为：", outlier.min())


# # 菜品售价根据箱线图识别异常值

# In[17]:


import matplotlib.pyplot as plt

plt.figure(figsize = (10, 8))
p = plt.boxplot(detail["counts"].values, notch = True) # 画出箱线图
outlier1 = p["fliers"][0].get_ydata() # fliers 为异常值的标签
plt.savefig("../tmp/菜品异常数据识别.png")
plt.show()
print("销售量数据异常值个数为：", len(outlier1))
print("销售量数据异常的最大值为：", max(outlier1))
print("销售量数据异常的最小值为：", min(outlier1))


# # 订单详情表的样本去重与特征去重

# In[4]:


import pandas as pd

detail = pd.read_csv("../data/detail.csv", index_col = 0, encoding = "gbk")
print("进行去重操作前订单详情表的形状为：", detail.shape)

# 样本去重

detail.drop_duplicates(inplace = True)

# 特征去重

def FeatureEquals(df):
    # 定义求取特征是否完全相同的矩阵的函数
    dfEquals = pd.DataFrame([], columns = df.columns, index = df.columns)
    for i in df.columns:
        for j in df.columns:
            dfEquals.loc[i, j] = df.loc[:, i].equals(df.loc[:, j])
    return dfEquals
detEquals = FeatureEquals(detail) # 应用上述函数

# 遍历所有数据

lenDet = detEquals.shape[0]
dupCol = []
for k in range(lenDet):
    for l in range(k + 1, lenDet):
        if detEquals.iloc[k, l] & (detEquals.columns[l] not in dupCol):
            dupCol.append(detEquals.columns[l])

# 删除重复列

detail.drop(dupCol, axis = 1, inplace = True)
print("进行去重操作后订单详情表的形状为：", detail.shape)


# # 订单详情表的缺失值检测与处理

# In[5]:


# 统计各个特征的缺失率

naRate = (detail.isnull().sum() / detail.shape[0] * 100).astype("str") + "%"
print("detail 每个特征缺失的率为：\n", naRate)

# 删除全部数据均为缺失的列

detail.dropna(axis = 1, how = "all", inplace = True)
print("经过缺失值处理后订单详情表各特征缺失值的数目为：\n", detail.isnull().sum())


# # 订单详情表异常值检测与处理

# In[6]:


# 定义异常值识别与处理函数

def outRange(Ser1):
    QL = Ser1.quantile(0.25)
    QU = Ser1.quantile(0.75)
    IQR = QU - QL
    Ser1.loc[Ser1 > (QU + 1.5 * IQR)] = QU
    Ser1.loc[Ser1 < (QL - 1.5 * IQR)] = QL
    return Ser1

# 处理销售量和售价的异常值

detail["counts"] = outRange(detail["counts"])
detail["amounts"] = outRange(detail["amounts"])

# 查看处理后的销售量，售价的最小值、最大值

print("销售量最小值为：", detail["counts"].min())
print("销售量最大值为：", detail["counts"].max())
print("售价最小值为：", detail["amounts"].min())
print("售价最大值为：", detail["amounts"].max())


# # 离差标准化示例

# In[8]:


import pandas as pd
import numpy as np

detail = pd.read_csv("../data/detail.csv", index_col = 0, encoding = "gbk")

# 自定义离差标准化函数

def MinMaxScale(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data

# 对菜品订单表售价和销量做离差标准化

data1 = MinMaxScale(detail["counts"])
data2 = MinMaxScale(detail["amounts"])
data3 = pd.concat([data1, data2], axis = 1)
print("离差标准化之前销量和售价数据为：\n", detail[["counts", "amounts"]].head())
print("离差标准化之后销量和售价数据为：\n", data3.head())


# # 标准差标准化示例

# In[9]:


# 自定义标准差标准化函数

def StandardScaler(data):
    data = (data - data.mean()) / data.std()
    return data

# 菜品订单表售价和销量做标准化

data4 = StandardScaler(detail["counts"])
data5 = StandardScaler(detail["amounts"])
data6 = pd.concat([data4, data5], axis = 1)
print("标准差标准化之前销量和售价数据为：\n", detail[["counts", "amounts"]].head())
print("标准差标准化之后销量和售价数据为：\n", data6.head())


# # 小数定标标准化示例

# In[10]:


# 自定义小数定标标准化函数

def DecimalScaler(data):
    data = data / 10 ** np.ceil(np.log10(data.abs().max()))
    return data

# 对菜品订单表售价和销量做标准化

data7 = DecimalScaler(detail["counts"])
data8 = DecimalScaler(detail["amounts"])
data9 = pd.concat([data7, data8], axis = 1)
print("小数定标标准化之前的销量和售价数据：\n", detail[["counts", "amounts"]].head())
print("小数定标标准化之后的销量和售价数据：\n", data9.head())


# # 对订单详情表中的数值型数据做标准差标准化

# In[14]:


# 自定义标准差标准化函数

def StandardScaler(data):
    data = (data - data.mean()) / data.std()
    return data

# 对菜品订单表售价和销量做标准化

data4 = StandardScaler(detail["counts"])
data5 = StandardScaler(detail["amounts"])
data6 = pd.concat([data4, data5], axis = 1)
print("标准差标准化之后售价和销量数据为：\n", data6.head(10))


# # get_dummies 函数的参数及说明

# In[15]:


# get_dummies 函数：

# pandas.get_dummies(data, prefix = None, prefix_sep = "_", dummy_na = False, columns = None, sparse = False, drop_first = False)

# data             接收 array、DataFrame 或者 Series。表示需要哑变量处理的数据。无默认

# prefix           接收 string、string 的列表或者 string 的 dict。表示哑变量化后列名的前缀。默认为 None

# prefix_sep       接收 string。表示前缀的连接符。默认为 "_"

# dummy_na         接收 boolean。表示是否为 NaN 值添加一列。默认为 False

# columns          接收类似 list 的数据。表示 DataFrame 中需要编码的列名。默认为 None，表示对所有 object 和 category 类型进行编码

# sparse           接收 boolean。表示虚拟列是否是稀疏的。默认为 False

# drop_first       接收 boolean。表示是否通过从 k 个分类级别中删除第一级来获得 k-1 个分类级别。默认为 False


# # 哑变量处理示例

# In[17]:


import pandas as pd
import numpy as np

detail = pd.read_csv("../data/detail.csv", encoding = "gbk")
data = detail.loc[0:5, "dishes_name"] # 抽取部分数据做演示
print("哑变量处理前的数据为：\n", data)
print("哑变量处理后的数据为：\n", pd.get_dummies(data))

