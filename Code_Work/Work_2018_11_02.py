
# coding: utf-8

# # 读取并查看 P2P 网络贷款数据主表的基本信息

# In[7]:


import pandas as pd

master_data = pd.read_csv("../data/Training_Master.csv", sep = ",", encoding = "gbk")

# 使用 ndim、shape、memory_usage 属性分别查看维度、大小、占用内存信息

print("P2P网络贷款数据主表的维度为：", master_data.ndim)
print("P2P网络贷款数据主表的元素形状（大小）为：", master_data.shape)
print("P2P网络贷款数据主表的占用内存信息为：\n", master_data.memory_usage)


# In[8]:


# 使用 describe 方法进行描述性统计

print("P2P网络贷款数据主表的描述性统计结果为：", master_data.describe())


# In[9]:


# 剔除值相同或全为空的列

# 定义一个函数剔除全为空值的列和标准差为 0 的列

def dropNullStd(data):
    beforelen = data.shape[1]
    colisNull = data.describe().loc["count"] == 0
    for i in range(len(colisNull)):
        if colisNull[i]:
            data.drop(colisNull.index[i], axis = 1, inplace = True)
    
    stdisZero = data.describe().loc["std"] == 0
    for i in range(len(stdisZero)):
        if stdisZero[i]:
            data.drop(stdisZero.index[i], axis = 1, inplace = True)
    afterlen = data.shape[1]
    print("剔除的列的数目为：", beforelen - afterlen)
    print("剔除后数据的形状为：", data.shape)

dropNullStd(master_data)


# # 读取 mtcars 数据集

# In[1]:


import pandas as pd

mtcars_data = pd.read_csv("../data/mtcars.csv", sep = ",", encoding = "gbk")

# 查看mtcars数据集的维度、大小等信息

print("mtcars数据集的维度为：", mtcars_data.ndim)
print("mtcars数据集的元素形状（大小）为：", mtcars_data.shape)
print("mtcars数据集的元素个数为：", mtcars_data.size)
print("mtcars数据集的占用内存信息为：\n", mtcars_data.memory_usage)


# In[7]:


# 使用describe方法对整个mtcars数据集进行描述性统计

print("mtcars数据集的描述性统计结果为：\n", mtcars_data.describe())


# In[6]:


# 计算不同cyl(汽缸数)、carb(化油器)对应的mpg(油耗)和hp(马力)的均值

mtcars_data_group = mtcars_data[["cyl", "carb", "mpg", "hp"]].groupby(by = ["cyl", "carb"])
print("不同 cyl (汽缸数)、carb (化油器)对应的 mpg (油耗)和 hp (马力)的均值结果为：\n", mtcars_data_group.mean())

