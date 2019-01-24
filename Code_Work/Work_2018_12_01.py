
# coding: utf-8

# # 插补用户用电量数据缺失值

# In[1]:


import pandas as pd
import numpy as np
from scipy.interpolate import lagrange

# 读取 missing_data.csv 表中的数据

missing_data = pd.read_excel("../data/missing_data.xls", header = None)

# 查询缺失值所在位置

print("缺失值所在位置的索引为：")
for i in missing_data.columns:
    for j in range(len(missing_data)):
        if (missing_data[i].isnull())[j]:
            print("(%d, %d)" %(j, i))

# 使用 Scipy 库中的 interpolate 模块中的 lagrange 对数据进行 lagrange 插值

def ployinterp_column(s, n): # s 为列向量，n 为被插值的位置
    y = s
    y = y[y.notnull()] # 剔除空值
    return lagrange(y.index, list(y))(n)

print("插值之前的数据为：\n", missing_data)
for i in missing_data.columns:
    for j in range(len(missing_data)):
        if (missing_data[i].isnull())[j]:
            missing_data[i][j] = ployinterp_column(missing_data[i], j)
print("插值之后的数据为：\n", missing_data)

# 查看数据中是否存在缺失值，若不存在则说明插值成功
print("数据表每列数据缺失值的个数为：")
print(missing_data.isnull().sum())


# # 合并线损、用电量趋势与线路告警数据

# In[2]:


import pandas as pd

# 读取 ele_loss.csv 和 alarm.csv 表

ele_loss_data = pd.read_csv("../data/ele_loss.csv", sep = ",", encoding = "gbk")
alarm_data = pd.read_csv("../data/alarm.csv", sep = ",", encoding = "gbk")

# 查看两表的形状

print("ele_loss 表的形状为：", ele_loss_data.shape)
print("alarm_data 表的形状为：", alarm_data.shape)

# 以 ID 和 date 两个键值作为主键进行内连接

ele_loss_alarm_data = pd.merge(ele_loss_data, alarm_data, left_on = ["ID", "date"], right_on = ["ID", "date"])

# 查看合并后的数据

print(ele_loss_alarm_data)


# # 标准化建模专家样本数据

# In[3]:


import pandas as pd

# 读取 model.csv 数据

model_data = pd.read_excel("../data/model.xls", sep = ",", encoding = "gbk")

# 定义标准差标准化函数

def StandardScaler(data):
    data = (data - data.mean()) / data.std()
    return data

# 使用函数分别对 3 列数据进行标准化

cel_data_0 = StandardScaler(model_data["电量趋势下降指标"])
cel_data_1 = StandardScaler(model_data["线损指标"])
cel_data_2 = StandardScaler(model_data["告警类指标"])

data = pd.concat([cel_data_0, cel_data_1, cel_data_2], axis = 1)

# 查看标准化后的数据

print("标准差标准化之后的电量趋势下降指标、线损指标、告警类指标的数据为：\n", data)

