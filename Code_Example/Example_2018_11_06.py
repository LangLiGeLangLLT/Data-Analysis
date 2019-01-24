
# coding: utf-8

# # 转换字符串时间为标准时间

# In[1]:


import pandas as pd

order = pd.read_table("../data/meal_order_info.csv", sep = ",", encoding = "gbk")
print("进行转换前订单信息表 lock_time 的类型为：", order["lock_time"].dtypes)
order["lock_time"] = pd.to_datetime(order["lock_time"])
print("进行转换后订单信息表 lock_time 的类型为：", order["lock_time"].dtypes)


# # Timestamp 的最小时间和最大时间

# In[2]:


print("最小时间为：", pd.Timestamp.min)


# In[3]:


print("最大时间为：", pd.Timestamp.max)


# # 时间字符串转换为 DatetimeIndex 和 PeriodIndex

# In[4]:


dateIndex = pd.DatetimeIndex(order["lock_time"])
print("转换为 DatetimeIndex 后数据的类型为：\n", type(dateIndex))


# In[5]:


periodIndex = pd.PeriodIndex(order["lock_time"], freq = "S")
print("转换为 PeriodIndex 后数据的类型为：\n", type(periodIndex))


# # 提取 datetime 数据中的时间序列数据

# In[6]:


year1 = [i.year for i in order["lock_time"]]
print("lock_time 中的年份数据前 5 个为：", year1[:5])
month1 = [i.month for i in order["lock_time"]]
print("lock_time 中的月份数据前 5 个为：", month1[:5])
day1 = [i.day for i in order["lock_time"]]
print("lock_time 中的日期数据前 5 个为：", day1[:5])
weekday1 = [i.weekday_name for i in order["lock_time"]]
print("lock_time 的星期名称数据前 5 个为：", weekday1[:5])


# # 提取 DatetimeIndex 和 PeriodIndex 中的数据

# In[7]:


print("dateIndex 中的星期名称数据前 5 个为：\n", dateIndex.weekday_name[:5])
print("periodIndex 中的星期标号数据前 5 个为：", periodIndex.weekday[:5])


# # 使用 Timedelta 实现时间数据的加运算

# In[8]:


# 将 lock_time 数据向后平移一天

time1 = order["lock_time"] + pd.Timedelta(days = 1)
print("lock_time 加上一天前前 5 行数据为：\n", order["lock_time"][:5])
print("lock_time 加上一天前前 5 行数据为：\n", time1[:5])


# # 使用 Timedelta 实现书剑数据的减运算

# In[9]:


timeDelta = order["lock_time"] - pd.to_datetime("2017-1-1")
print("lock_time 减去 2017 年 1 月 1 日 0 点 0 时 0 分后的数据：\n", timeDelta[:5])
print("lock_time 减去 time1 后的数据类型为：", timeDelta.dtypes)


# # 订单信息表时间数据转换

# In[10]:


import pandas as pd

order = pd.read_table("../data/meal_order_info.csv", sep = ",", encoding = "gbk")
order["use_start_time"] = pd.to_datetime(order["use_start_time"])
order["lock_time"] = pd.to_datetime(order["lock_time"])
print("进行转换后订单信息表 use_start_time 和 lock_time 的类型为：\n", order[["use_start_time", "lock_time"]].dtypes)


# # 订单信息表时间信息提取

# In[12]:


year = [i.year for i in order["lock_time"]] # 提取年份信息
month = [i.month for i in order["lock_time"]] # 提取月份信息
day = [i.day for i in order["lock_time"]] # 提取日期信息
week = [i.week for i in order["lock_time"]] # 提取周信息
weekday = [i.weekday() for i in order["lock_time"]] # 提取星期信息

# 提取星期名称信息

weekname = [i.weekday_name for i in order["lock_time"]]
print("订单详情表中的前 5 条数据的年份信息为：", year[:5])
print("订单详情表中的前 5 条数据的月份信息为：", month[:5])
print("订单详情表中的前 5 条数据的日期信息为：", day[:5])
print("订单详情表中的前 5 条数据的周信息为：", week[:5])
print("订单详情表中的前 5 条数据的星期信息为：", weekday[:5])
print("订单详情表中的前 5 条数据的星期名称信息为：", weekname[:5])


# # 查看订单信息表时间统计信息

# In[13]:


timemin = order["lock_time"].min()
timemax = order["lock_time"].max()
print("订单最早的时间为：", timemin)
print("订单最晚的时间为：", timemax)
print("订单持续的时间为：", timemax - timemin)


# In[14]:


chekTime = order["lock_time"] - order["use_start_time"]
print("平均点餐时间为：", chekTime.mean())
print("最短点餐时间为：", chekTime.min())
print("最长点餐时间为：", chekTime.max())


# # 对菜品订单详情表依据订单编号分组

# In[15]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
detail = pd.read_sql_table("meal_order_detail1", con = engine)
detailGroup = detail[["order_id", "counts", "amounts"]].groupby(by = "order_id")
print("分组后的订单详情表为：", detailGroup)


# # GroupBy 类求均值、标准差、中位数

# In[16]:


print("订单详情表分组后前 5 组每组的均值为：\n", detailGroup.mean().head())


# In[17]:


print("订单详情表分组后前 5 组每组的标准差为：\n", detailGroup.std().head())


# In[18]:


print("订单详情表分组后前 5 组每组的大小为：\n", detailGroup.size().head())


# # 使用 agg 求出当前数据对应的统计量

# In[19]:


print("订单详情表的菜品销量与售价的和与均值为：\n", detail[["counts", "amounts"]].agg([np.sum, np.mean]))


# # 使用 agg 分别求字段的不同统计量

# In[20]:


print("订单详情表的菜品销量总和与售价的均值为：\n", detail.agg({"counts":np.sum, "amounts":np.mean}))


# # 使用 agg 方法求不同字段的不同数目统计量

# In[21]:


print("菜品订单详情表的菜品销量总和与均值为：\n", detail.agg({"counts":np.sum, "amounts":[np.mean, np.sum]}))


# # 在 agg 方法中使用自定义函数

# In[22]:


# 自定义函数求两倍的和

def DoubleSum(data):
    s = data.sum() * 2
    return s

print("菜品订单详情表的菜品销量两倍总和为：\n", detail.agg({"counts":DoubleSum}, axis = 0))


# # agg 方法中使用的自定义函数含 NumPy 中的函数

# In[23]:


# 自定义函数求两倍的和

def DoubleSum1(data):
    s = np.sum(data) * 2
    return s

print("订单详情表的菜品销量两倍总和为：\n", detail.agg({"counts":DoubleSum1}, axis = 0).head())


# In[24]:


print("订单详情表的菜品销量与售价的和的两倍为：\n", detail[["counts", "amounts"]].agg(DoubleSum1))


# # 使用 agg 方法做简单的聚合

# In[25]:


print("订单详情表分组后前 3 组每组的均值为：\n", detailGroup.agg(np.mean).head(3))


# In[26]:


print("订单详情表分组后前 3 组每组的标准差为：\n", detailGroup.agg(np.std).head(3))


# # 使用 agg 方法对分组数据使用不同的聚合函数

# In[27]:


print("订单详情分组前 3 组每组菜品总数和售价均值为：\n",detailGroup.agg({"counts":np.sum, "amounts":np.mean}).head(3))


# # apply 方法的基本用法

# In[28]:


print("订单详情表的菜品销量与售价的均值为：\n", detail[["counts", "amounts"]].apply(np.mean))


# # 使用 apply 方法进行聚合操作

# In[29]:


print("订单详情表分组后前 3 组每组的均值为：\n", detailGroup.apply(np.mean).head(3))


# In[30]:


print("订单详情表分组后前 3 组每组的标准差为：\n", detailGroup.apply(np.std).head(3))


# # 使用 transform 方法将销量和售价翻倍

# In[34]:


print("订单详情表的菜品销量与售价的两倍为：\n", detail[["counts", "amounts"]].transform(lambda x:x * 2).head(4))


# # 使用 transform 实现组内离差标准化

# In[36]:


print("订单详情表分组后实现组内离差标准化后前 5 行为：\n", detailGroup.transform(lambda x:(x.mean() - x.min()) / (x.max() - x.min())).head())


# # 订单详情表按照日期分组

# In[37]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
detail = pd.read_sql_table("meal_order_detail1", con = engine)
detail["place_order_time"] = pd.to_datetime(detail["place_order_time"])
detail["date"] = [i.date() for i in detail["place_order_time"]]
detailGroup = detail[["date", "counts", "amounts"]].groupby(by = "date")
print("订单详情表前 5 组每组的数目为：\n", detailGroup.size().head())


# # 求分组后的订单详情表每日菜品销售的均价、中位数

# In[38]:


dayMean = detailGroup.agg({"amounts":np.mean})
print("订单详情表前 5 组单日菜品销售均价为：\n", dayMean.head())


# In[41]:


dayMedian = detailGroup.agg({"amounts":np.median})
print("订单详情表前 5 组单日菜品售价中位数为：\n", dayMedian.head())


# # 求取订单详情表中单日菜品总销量

# In[42]:


daySaleSum = detailGroup.apply(np.sum)["counts"]
print("订单详情表前 5 组单日菜品售出数目为：\n", daySaleSum.head())

