
# coding: utf-8

# # 订单详情表的 4 个基本属性

# In[1]:


from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8") # 创建数据库连接
detail = pd.read_sql_table("meal_order_detail1", con = engine)
print("订单详情表的索引为：", detail.index)


# In[2]:


print("订单详情表的所有值为：\n", detail.values)


# In[3]:


print("订单详情表的列名为：\n", detail.columns)


# In[4]:


print("订单详情表的数据类型为：\n", detail.dtypes)


# # size、ndim 和 shape 属性的使用

# In[5]:


# 查看 DataFrame 的元素个数

print("订单详情表的元素个数为：", detail.size)
print("订单详情表的维度数为：", detail.ndim) # 查看 DataFrame 的维度数
print("订单详情表的形状为：", detail.shape) # 查看 DataFrame 的形状


# # 使用 T 属性进行转置

# In[6]:


print("订单详情表转置前形状为：", detail.shape)
print("订单详情表转置后形状为：", detail.T.shape)


# # 使用字典访问内部数据的方式访问 DataFrame 单列数据

# In[7]:


# 使用字典访问的方式取出 orderInfo 中的某一列

order_id = detail["order_id"]
print("订单详情表中的 order_id 的形状为：\n", order_id.shape)


# # 使用访问属性的方式访问 DataFrame 单列数据

# In[8]:


# 使用访问属性的方式取出 orderInfo 中的菜品名称列

dishes_name = detail.dishes_name
print("订单详情表中的 dishes_name 的形状为：", dishes_name.shape)


# # DataFrame 单列多行数据获取

# In[9]:


dishes_name5 = detail["dishes_name"][:5]
print("订单详情表中的 dishes_name 前 5 个元素为：\n", dishes_name5)


# # 访问 DataFrame 多列的多行数据

# In[10]:


orderDish = detail[["order_id","dishes_name"]][:5]
print("订单详情表中的 order_id 和 dishes_name 前 5 个元素为：\n", orderDish)


# # 访问 DataFrame 多行数据

# In[11]:


order5 = detail[:][1:6]
print("订单详情表的 1~6 行元素为：\n", order5)


# # 使用 DataFrame 的 head 和 tail 方法获取多行数据

# In[12]:


print("订单详情表中前 5 行数据为：\n", detail.head())


# In[13]:


print("订单详情表中后 5 行元素为：\n", detail.tail())


# # 使用 loc 和 iloc 实现单列切片

# In[14]:


dishes_name1 = detail.loc[:, "dishes_name"]
print("使用 loc 提取 dishes_name 列的 size 为：", dishes_name.size)


# In[15]:


dishes_name2 = detail.iloc[:, 3]
print("使用 iloc 提取第 3 列的 size 为：", dishes_name2.size)


# # 使用 loc、iloc 实现多列切片

# In[16]:


orderDish1 = detail.loc[:, ["order_id", "dishes_name"]]
print("使用 loc 提取 order_id 和 dishes_name 列的 size 为：", orderDish1.size)


# In[2]:


orderDish2 = detail.iloc[:, [1, 3]]
print("使用iloc提取第 1 和第 3 列的 size 为：", orderDish2.size)


# # 使用 loc、iloc 实现花式切片

# In[17]:


print("列名为 order_id 和 dishes_name 的行名为 3 的数据为：\n", detail.loc[3, ["order_id", "dishes_name"]])


# In[18]:


print("列名为 order_id 和 dishes_name 行名为 2,3,4,5,6 的数据为：\n", detail.loc[2:6, ["order_id", "dishes_name"]])


# In[19]:


print("列位置 1 和 3，行位置为 3 的数据为：\n", detail.iloc[3, [1, 3]])


# In[21]:


print("列位置为 1 和 3，行位置为 2,3,4,5,6 的数据为：\n", detail.iloc[2:7, [1, 3]])


# # 使用 loc 和 iloc 实现条件切片

# In[22]:


# loc内部传入表达式

print("detail 中 order_id 为 458 的 dishes_name 为：\n", detail.loc[detail["order_id"] == "458", ["order_id", "dishes_name"]])


# In[23]:


print("detail 中 order_id 为 458 的第 1、5 列数据为：\n", detail.iloc[detail["order_id"] == "458", [1, 5]])


# # 使用 iloc 实现条件切片

# In[24]:


print("detail 中 order_id 为 458 的第 1、5 列数据为：\n", detail.iloc[(detail["order_id"] == "458").values, [1, 5]])


# # 使用 loc、iloc、ix 实现切片比较

# In[25]:


print("列名为 dishes_name 行名为 2,3,4,5,6 的数据为：\n", detail.loc[2:6, "dishes_name"])


# In[26]:


print("列位置为 5，行位置为 2~6 的数据为：\n", detail.iloc[2:6, 5])


# In[28]:


print("列位置为 5，行名为 2~6 的数据为：\n", detail.ix[2:6, 5])


# # 更改 DataFrame 中的数据

# In[29]:


# 将 order_id 为 458 的变换为 45800

detail.loc[detail["order_id"] == "458", "order_id"] = "45800"
print("更改后 detail 中 order_id 为 458 的 order_id 为：\n", detail.loc[detail["order_id"] == "458", "order_id"])
print("更改后 detail 中 order_id为 45800 的 order_id 为：\n", detail.loc[detail["order_id"] == "45800", "order_id"])


# # 为 DataFrame 新增一列非定值

# In[2]:


detail["payment"] = detail["counts"] * detail["amounts"]
print("detail 新增列 payment 的前 5 行为：\n", detail["payment"].head())


# # DataFrame 新增一列定值

# In[3]:


detail["pay_way"] = "现金支付"
print("detail 新增列 pay_way 的前 5 行为：\n", detail["pay_way"].head())


# # 删除 DataFrame 某列

# In[4]:


print("删除 pay_way 前 detail 的列索引为：\n", detail.columns)
detail.drop(labels = "pay_way", axis = 1, inplace = True)
print("删除 pay_way 后 detail 的列索引为：\n", detail.columns)


# # 删除 DataFrame 某几行

# In[5]:


print("删除 1~10 行前 detail 的长度为：", len(detail))
detail.drop(labels = range(1, 11), axis = 0, inplace = True)
print("删除 1~10 行后 detail 的长度为：", len(detail))


# # 代码使用 np.mean 函数计算平均价格

# In[6]:


import numpy as np

print("订单详情表中 amount (价格)的平均值为：", np.mean(detail["amounts"]))


# # 通过 pandas 实现销量和价格的协方差矩阵计算

# In[7]:


print("订单详情表中 amount (价格)的平均值为：", detail["amounts"].mean())


# # 使用 describe 方法实现数值型特征的描述性统计

# In[8]:


print("订单详情表 counts 和 amounts 两列的描述性统计为：\n", detail[["counts", "amounts"]].describe())


# # 对菜品名称频数统计

# In[9]:


print("订单详情表 dishes_name 频数统计结果前 10 为：\n", detail["dishes_name"].value_counts()[0:10])


# # 将 object 数据强制转换为 category 类型

# In[10]:


detail["dishes_name"] = detail["dishes_name"].astype("category")
print("订单详情表 dishes_name 列转变数据类型后为：", detail["dishes_name"].dtype)


# # category 类型特征的描述性统计

# In[11]:


print("订单详情表 dishes_name 的描述统计结果为：\n", detail["dishes_name"].describe())


# # 查看餐饮数据基本信息

# In[1]:


from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
detail = pd.read_sql_table("meal_order_detail1", con = engine)
order = pd.read_table("../data/meal_order_info.csv", sep = ",", encoding = "gbk")
user = pd.read_excel("../data/users.xlsx")
print("订单详情表的维度为：", detail.ndim)
print("订单详情表的维度为：", order.ndim)
print("订单详情表的维度为：", user.ndim)


# In[2]:


print("订单详情表的形状为：", detail.shape)
print("订单信息表的形状为：", order.shape)
print("客户信息表的形状为：", user.shape)


# In[3]:


print("订单详情表的元素个数为：", detail.size)
print("订单信息表的元素个数为：", order.size)
print("客户信息表的元素个数为：", user.size)


# # 餐饮菜品销量的描述性统计

# In[4]:


print("订单详情表 counts 和 amounts 两列的描述性统计为：\n", detail.loc[:, ["counts", "amounts"]].describe())


# In[5]:


detail["order_id"] = detail["order_id"].astype("category")
detail["dishes_name"] = detail["dishes_name"].astype("category")
print("订单信息表 order_id (订单编号)与 dishes_name (菜品名称)的描述性统计结果为：\n", detail[["order_id", "dishes_name"]].describe())


# # 剔除餐饮菜品中整列为空或者取值完全相同的列

# In[6]:


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
            data.drop(stdisZero.index[i], axis = 1, inplace=True)
    afterlen = data.shape[1]
    print("剔除的列的数目为：", beforelen - afterlen)
    print("剔除后数据的形状为：", data.shape)

dropNullStd(detail)


# In[7]:


# 使用 dropNullStd 函数对订单信息表操作

dropNullStd(order)


# In[8]:


# 使用 dropNullStd 函数对客户信息表操作

dropNullStd(user)

