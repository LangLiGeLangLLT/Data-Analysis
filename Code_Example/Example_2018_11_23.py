
# coding: utf-8

# # merge 函数的参数及其说明

# In[1]:


# merge 函数：

# pandas.merge(left, right, how = "inner", on = None, left_on = None, right_on = None, left_index = False, right_index = False,
#              sort = False, suffixes = ("_x", "_y"), copy = True, indicator = False)

# left               接收 DataFrame 或 Series。表示要添加的新数据 1。无默认

# right              接收 DataFrame 或 Series。表示要添加的新数据 2。无默认

# how                接收 inner、outer、left、right。表示数据的连接方式。默认为 inner

# on                 接收 string 或 sequence。表示两个数据合并的主键（必须一致）。默认为 None

# left_on            接收 string 或 sequence。表示 left 参数接收数据用于合并的主键。默认为 None

# right_on           接收 string 或 sequence。表示 right 参数接收数据用于合并的主键。默认为 None

# left_index         接收 boolean。表示是否将 left 参数接收数据的 index 作为连接主键。默认为 False

# right_index        接收 boolean。表示是否将 right 参数接收数据的 index 作为连接主键。默认为 False

# sort               接收 boolean。表示是否根据连接键对合并后的数据进行排序。默认为 False

# suffixes           表示用于追加到 left 和 right 参数接收数据列名相同时的后缀。默认为 ("_x", "_y")


# # 使用 merge 函数合并数据表

# In[2]:


import numpy as np
import pandas as pd
from sqlalchemy import create_engine

conn = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
detail1 = pd.read_sql("meal_order_detail1", conn)
order = pd.read_csv("../data/meal_order_info.csv", sep = ",", encoding = "gb18030") # 读取订单信息表

# 将 info_id 转换为字符串格式，为合并做准备

order["info_id"] = order["info_id"].astype("str")
# detail1["order_id"] = detail1["order_id"].astype("int64")

# 订单详情表和订单信息表都有订单编号
# 在订单详情表中为 order_id，在订单信息表中为 info_id

order_detail = pd.merge(detail1, order, left_on = "order_id", right_on = "info_id")
print("detail1 订单详情表的原始形状为：", detail1.shape)
print("order 订单信息表的原始形状为：", order.shape)
print("订单详情表和订单信息表主键合并后的形状为：", order_detail.shape)


# # join 方法的参数及其说明

# In[3]:


# join 函数：

# pandas.DataFrame.join(self, other, on = None, how = "left", lsuffix = "", rsuffix = "", sort = False)

# other               接收 DataFrame、Series 或者包含了多个 DataFrame 的 list。表示参与连接的其他 DataFrame。无默认

# on                  接收列名或者包含列名的 list 或 tuple。表示用于连接的列名。默认为 None

# how                 接收待定 string。inner 代表内连接；outer 代表外连接；left 和 right 分别代表左连接和右连接。默认为 inner

# lsuffix             接收 string。表示用于追加到左侧重叠列名的尾缀。无默认

# rsuffix             接收 string。表示用于追加到右侧重叠列名的尾缀。无默认

# sort                接收 boolean。根据连接键对合并后的数据进行排序，默认为 False


# # 使用 join 方法实现主键合并

# In[4]:


order.rename({"info_id":"order_id"}, inplace = True)
detail1["order_id"] = detail1["order_id"].astype("int64")
order_detail1 = detail1.join(order, on = "order_id", rsuffix = "1")
print("订单详情表和订单信息表 join 合并后的形状为：", order_detail1.shape)


# # combine_first 方法常用参数及其说明

# In[5]:


# combine_first 函数：

# pandas.DataFrame.combine_first(other)

# other              接收 DataFrame。表示参与重叠合并的另一个 DataFrame。无默认


# # 重叠合并

# In[6]:


# 建立两个字典，除了 ID 外，别的特征互补

dict1 = {"ID":[1, 2, 3, 4, 5, 6, 7, 8, 9],
         "System":["win10", "win10", np.nan, "win10", np.nan, np.nan, "win7", "win7", "win8"],
         "cpu":["i7", "i5", np.nan, "i7", np.nan, np.nan, "i5", "i5", "i3"]}
dict2 = {"ID":[1, 2, 3, 4, 5, 6, 7, 8, 9],
         "System":[np.nan, np.nan, "win7", np.nan, "win8", "win7", np.nan, np.nan, np.nan],
         "cpu":[np.nan, np.nan, "i3", np.nan, "i7", "i5", np.nan, np.nan, np.nan]}

# 转换两个字典为 DataFrame

df5 = pd.DataFrame(dict1)
df6 = pd.DataFrame(dict2)
print("经过重叠合并后的数据为：\n", df5.combine_first(df6))


# # 将多张菜品订单详情表纵向合并

# In[7]:


import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# 创建数据库连接

conn = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")

# 读取数据

detail1 = pd.read_sql("meal_order_detail1", conn)
detail2 = pd.read_sql("meal_order_detail2", conn)
detail3 = pd.read_sql("meal_order_detail3", conn)

# 纵向堆叠 3 张表

detail = detail1.append(detail2)
detail = detail.append(detail3)
print("3 张订单详情表合并后的形状为：", detail.shape)


# # 订单详情表、订单信息表、客户信息表主键合并

# In[8]:


order = pd.read_csv("../data/meal_order_info.csv", sep = ",", encoding = "gb18030") # 读取订单信息表
user = pd.read_excel("../data/users_info.xlsx") # 读取客户信息表

# 数据类型转换，存储部分数据

order["info_id"] = order["info_id"].astype("str")
order["emp_id"] = order["emp_id"].astype("str")
user["USER_ID"] = user["USER_ID"].astype("str")
data = pd.merge(detail, order, left_on = ["order_id", "emp_id"], right_on = ["info_id", "emp_id"])
data = pd.merge(data, user, left_on = "emp_id", right_on = "USER_ID", how = "inner")
print("3 张表数据主键合并后的大小为：", data.shape)


# # 利用 list 去重

# In[9]:


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
print("去重前菜品总数为：",len(dishes))
dish = delRep(dishes) # 使用自定义的去重函数去重
print("方法一去重后菜品总数为：", len(dish))


# # 利用 set 的特性去重

# In[10]:


# 方法二

print("去重前菜品总数为：", len(dishes))
dish_set = set(dishes) # 利用 set 的特性去重
print("方法二去重后的菜品总数为：", len(dish_set))


# # drop_duplicates 方法的常用参数及其说明

# In[11]:


# drop_duplicates 函数：

# pandas.DataFrame(Series).drop_duplicates(self, subset = None, keep = "first", inplace = False)

# subset               接收 string 或 sequence。表示进行去重的列。默认为 None，表示全部列

# keep                 接收待定 string。表示重复时保留第几个数据
#                      first：保留第一个
#                      last：保留最后一个
#                      false：只要有重复都不保留
#                      默认为first

# inplace              接收 boolean。表示是否在原表上进行操作。默认为 False


# # 使用 drop_duplicates 方法对菜品名称去重

# In[12]:


# 对 dishes_name 去重

dishes_name = detail["dishes_name"].drop_duplicates()
print("drop_duplicates 方法去重之后菜品总数为：", len(dishes_name))


# # 使用 drop_duplicates 方法对多列去重

# In[13]:


print("去重之前订单详情表的形状为：", detail.shape)
shapeDet = detail.drop_duplicates(subset = ["order_id", "emp_id"]).shape
print("依照订单编号，会员编号去重之后订单详情表大小为：", shapeDet)


# # 求出 counts 和 amounts 两列数据的 kendall 发相似度矩阵

# In[18]:


# 求取销量和售价的相似度

corrDet = detail[["counts", "amounts"]].corr(method = "kendall")
print("销量和售价的 kendall 法相似度矩阵为：\n", corrDet)


# # 求出 dishes_name、counts 和 amounts 这 3 个特征的 pearson 法相似度矩阵

# In[15]:


corrDet1 = detail[["dishes_name", "counts", "amounts"]].corr(method = "pearson")
print("菜品名称、销量和售价的 pearson 法相似度矩阵为：\n", corrDet1)


# # 使用 DataFrame.equals 方法去重

# In[16]:


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

# In[17]:


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

