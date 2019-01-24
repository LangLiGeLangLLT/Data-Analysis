
# coding: utf-8

# # 使用 pivot_table 函数创建透视表

# pivot_table 函数：
# 
# pandas.pivot_table(data, values = None, index = None, columns = None, aggfunc = "mean",
#   fill_value = None, margins = False, dropna = True, margins_name = "All")
# 
# data               接受 DataFrame。表示创建表的数据。无默认
# 
# values             接受 string。用于指定要聚合的数据字段名，默认使用全部数据。默认为 None
# 
# index              接受 string 或 list。表示行分组键。默认为 None
# 
# columns            接受 string 或 list。表示列分组键。默认为 None
# 
# aggfunc            接受 functions。表示聚合函数。默认为 mean
# 
# margins            接受 boolean。表示汇（Total）功能的开关，设置为 True 后，结果集中会出现名为 “ALL” 的行和列。默认为 True
# 
# dropna             接受 boolean。表示是否删掉全为 NaN 的列。默认为 False

# # 使用订单号作为透视表索引制作透视表

# In[2]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
detail = pd.read_sql_table("meal_order_detail1", con = engine)
detailPivot = pd.pivot_table(detail[["order_id", "counts", "amounts"]], index = "order_id")
print("以 order_id 作为分组键创建的订单透视表为：\n", detailPivot.head())


# # 修改聚合函数后的透视表

# In[3]:


detailPivot1 = pd.pivot_table(detail[["order_id", "counts", "amounts"]], index = "order_id", aggfunc = np.sum)
print("以 order_id 作为分组键创建的订单销量与售价总和透视表为：\n", detailPivot1.head())


# # 使用订单号和菜品名称作为索引的透视表

# In[4]:


detailPivot2 = pd.pivot_table(detail[["order_id", "dishes_name", "counts", "amounts"]], index = ["order_id", "dishes_name"], aggfunc = np.sum)
print("以 order_id 和 dishes_name 作为分组键创建的订单销量与售价总和透视表为：\n", detailPivot2.head())


# # 指定菜品名称为列分组键的透视表

# In[5]:


detailPivot2 = pd.pivot_table(detail[["order_id", "dishes_name", "counts", "amounts"]], index = "order_id", columns = "dishes_name", aggfunc = np.sum)
print("以 order_id 和 dishes_name 作为行列分组键创建的透视表前 5 行 4 列为：\n", detailPivot2.iloc[:5, :4])


# # 指定某些列制作透视表

# In[6]:


detailPivot4 = pd.pivot_table(detail[["order_id", "dishes_name", "counts", "amounts"]], index = "order_id", values = "counts", aggfunc = np.sum)
print("以 order_id 作为行分组键 counts 作为值创建的透视表前 5 行为：\n", detailPivot4.head())


# # 对透视表中的缺失值进行填充

# In[7]:


detailPivot5 = pd.pivot_table(detail[["order_id", "dishes_name", "counts", "amounts"]], index = "order_id", columns = "dishes_name", aggfunc = np.sum, fill_value = 0)
print("空值填0后以 order_id 和 dishes_name 为行列分组键创建透视表前 5 行 4 列为：\n", detailPivot5.iloc[:5, :4])


# # 在透视表中添加汇总数据

# In[8]:


detailPivot6 = pd.pivot_table(detail[["order_id", "dishes_name", "counts", "amounts"]], index = "order_id", columns = "dishes_name", aggfunc = np.sum, fill_value = 0, margins = True)
print("添加 margins 以后 order_id 和 dishes_name 为分组键的透视表前 5 行后 4 列为：\n", detailPivot6.iloc[:5, -4:])


# # 使用 crosstab 函数创建交叉表

# crosstab 函数：
# 
# pandas.crosstab(index, columns, values = None, rownames = None, colnames = None, aggfunc = None, margins = False, dropna = True, normalize = False)
# 
# index                       接受 string 或 list。表示行索引值。无默认
# 
# columns                     接受 string 或 list。表示列索引值。无默认
# 
# values                      接受 array。表示聚合数据。默认为 None
# 
# rownames                    表示行分组键名。无默认
# 
# colnames                    表示列分组键名。无默认
# 
# aggfunc                     接收 function。表示聚合函数。默认为 None
# 
# margins                     接收 boolean。默认为 True。表示汇总（Total）功能的开关，设置为 True 后，结果集中会出现名为“ALL”的行和列
# 
# dropna                      接受 boolean。表示是否删掉全为 NaN 的列。默认为 False
# 
# normalize                   接受 boolean。表示是否对值进行标准化。默认为 False

# # 使用 crosstab 函数制作交叉表

# In[9]:


detailCross = pd.crosstab(index = detail["order_id"], columns = detail["dishes_name"], values = detail["counts"], aggfunc = np.sum)
print("以 order_id 和 dishes_name 为分组键 counts 为值得透视表前 5 行 5 列为：\n", detailCross.iloc[:5, :5])


# # 订单详情表单日菜品成交总额与总数透视表

# In[10]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = pd.read_sql_table("meal_order_detail1", con = engine)
detail["place_order_time"] = pd.to_datetime(detail["place_order_time"])
detail["date"] = [i.date() for i in detail["place_order_time"]]
PivotDetail = pd.pivot_table(detail[["date", "dishes_name", "counts", "amounts"]], index = "date", aggfunc = np.sum, margins = True)
print("订单详情表单日菜品成交总额与总数透视表前 5 行为：\n", PivotDetail.head())


# # 订单详情表单个菜品单日成交总额透视表

# In[11]:


CrossDetail = pd.crosstab(index = detail["date"], columns = detail["dishes_name"], values = detail["amounts"], aggfunc = np.sum, margins = True)
print("订单详情表单个菜品单日成交总额交叉表后 5 行 5 列为：\n",CrossDetail.iloc[-5:, -5:])


# # 堆叠合并数据

# concat 函数：
# 
# pandas.concat(objs, axis = 0, join = "outer", join_axes = None, ignore_index = False, keys = None, levels = None, names = None, verify_integrity = False, copy = True)
# 
# objs                 接收多个 Series、DataFrame、Panel 的组合。表示参与连接的 pandas 对象的列表的组合。无默认
# 
# axis                 接收0或1。表示连接的轴向，默认为0
# 
# join                 接收 inner 或 outer。表示其他轴向上的索引是按交集（inner）还是并集（outer）进行合并。默认为 outer
# 
# join_axes            接收 Index 对象。表示用于其他 n-1 条轴的索引，不执行并集/交集运算
# 
# ignore_index         接收 boolean。表示是否不保留连接轴上的索引，产生一组新索引 range(total_length)。默认为 False
# 
# keys                 接收 sequence。表示与连接对象有关的值，用于形成连接轴向上的层次化索引。默认为 None
# 
# levels               接收包含多个 sequence 的 list。表示在指定 keys 参数后，指定用作层次化索引各级别上的索引。默认为 None
# 
# names                接收 list。表示在设置了 keys 和 levels 参数后，用于创建分层级别的名称。默认为 None
# 
# verify_integrity     接收 boolean。检查新连接的轴是否包含重复项。如果发现重复项，则引发异常。默认为 False

# # 索引完全相同时的横向堆叠

# In[20]:


import numpy as np
import pandas as pd
from sqlalchemy import create_engine

conn = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
detail1 = pd.read_sql("meal_order_detail1", conn)
df1 = detail1.iloc[:, :10] # 取出 detail 的前 10 列数据
df2 = detail1.iloc[:, 10:] # 取出 detail 的后 9 列数据
print("合并 df1 的大小为 %s，df2 的大小为 %s。" %(df1.shape, df2.shape))
print("內连接合并后的数据框大小为：", pd.concat([df1, df2], axis = 1, join = "inner").shape)
print("外连接合并后的数据框大小为：", pd.concat([df1, df2], axis = 1, join = "outer").shape)


# # 表名完全相同时的 concat 纵向堆叠

# In[21]:


df3 = detail1.iloc[:1500, :] # 取出 detail 前 1500 行数据
df4 = detail1.iloc[1500:, :] # 取出 detail 的 1500 后的数据
print("合并 df3 的大小为 %s，df4 的大小为 %s" %(df3.shape, df4.shape))
print("外连接纵向合并后的数据框大小为：", pd.concat([df3, df4], axis = 1, join = "outer").shape)
print("內连接纵向合并后的数据框大小为：", pd.concat([df3, df4], axis = 1, join = "inner").shape)


# # append 方法的基本语法

# append 函数：
# 
# pandas.DataFrame.append(self, other, ignore_index = False, verify_integrity = False)
# 
# other                    接收 DataFrame 或 Series。表示要添加的新数据。无默认
# 
# ignore_index             接收 boolean。如果输入 True，就会对新生成的 DataFrame 使用新的索引（自动产生），而忽略原来数据的索引。默认为 False
# 
# verify_integrity         接收 boolean。如果输入 True，那么当 ignore_index 为 False 时，会检查添加的数据索引是否冲突，如果冲突，则会添加失败。                          默认为 False

# # 使用 append 方法进行纵向表堆叠

# In[22]:


print("堆叠前 df3 的大小为 %s，df4 的大小为 %s" %(df3.shape, df4.shape))
print("append 纵向堆叠后的数据框大小为：", df3.append(df4).shape)


# # 主键合并数据

# merge 函数：
# 
# pandas.merge(left, right, how = "inner", on = None, left_on = None, right_on = None, left_index = False, right_index = False, sort = False, suffixes = ("_x", "_y"), copy = True, indicator = False)
# 
# left                 接收 DataFrame 或 Series。表示要添加的新数据1。无默认
# 
# right                接收 DataFrame 或 Series。表示要添加的新数据2。无默认
# 
# how                  接收 inner、outer、left、right。表示数据的连接方式。默认为 inner
# 
# on                   接收 string 或 sequence。表示两个数据合并的主键（必须一致）。默认为 None
# 
# left_on              接收 string 或 sequence。表示 left 参数接收数据用于合并的主键。默认为 None
# 
# right_on             接收 string 或 sequence。表示 right 参数接收数据用于合并的主键。默认为 None
# 
# left_index           接收 boolean。表示是否将 left 参数接收数据的 index 作为连接主键。默认为 False
# 
# right_index          接受 boolean。表示是否将 right 参数接收数据的 index 作为连接主键。默认为 False
# 
# sort                 接收 boolean。表示是否根据连接键对合并后的数据进行排序。默认为 False
# 
# suffixes             接受 tuple。表示用于追加到 left 和 right 参数接收数据列名相同时的后缀。默认为 ("_x", "_y")

# # 使用 merge 函数合并数据表

# In[42]:


order = pd.read_csv("../data/meal_order_info.csv", sep = ",", encoding = "gb18030") # 读取订单信息表

# 将 info_id 转换为字符串格式，为合并做准备

order["info_id"] = order["info_id"].astype("str")

# 订单详情表和订单信息表都有订单编号
# 在订单详情表中为 order_id，在订单信息表中为 info_id

order_detail = pd.merge(detail1, order, left_on = "order_id", right_on = "info_id")
print("detail1 订单详情表的原始形状为：", detail1.shape)
print("order 订单信息表的原始形状为：", order.shape)
print("订单详情表和订单信息表主键合并后的形状为：", order_detail.shape)


# # join 方法实现部分主键合并的功能

# join 函数：
# 
# pandas.DataFrame.join(self, other, on = None, how = "left", lsuffix = "", rsuffix = "", sort = False)
# 
# other          接收 DataFrame、Series 或者包含了多个 DataFrame 的 list。表示参与连接的其他 DataFrame。无默认
# 
# on             接收特定 string。inner 代表内连接；outer 代表外连接；left 和 right 分别代表左连接和右连接。默认为 inner
# 
# lsuffix        接收 string。表示用于追加到左侧重叠列名的尾缀。无默认
# 
# rsuffix        接收 string。表示用于追加到右侧重叠列名的尾缀。无默认
# 
# sort           接收 boolean。根据连接键对合并后的数据进行排序，默认为 False

# # 使用 join 方法实现主键合并

# In[47]:


order.rename({"info_id":"order_id"}, inplace = True)
order_detail1 = detail1.join(order, on = "order_id", rsuffix = "1")
print("订单详情表和订单信息表 join 合并后的形状为：", order_detail1.shape)


# # combine_first 方法的具体用法

# combine_first 函数：
# 
# pandas.DataFrame.combine_first(other)
# 
# other          接收 DataFrame。表示参与重叠合并的另一个 DataFrame。无默认

# In[30]:


# 建立两个字典，除了 ID 外，别的特征互补

dict1 = {"ID":[1, 2, 3, 4, 5, 6, 7, 8, 9], "System":["win10", "win10", np.nan, "win10", np.nan, np.nan, "win7", "win7", "win8"],
        "cpu":["i7", "i5", np.nan, "i7", np.nan, np.nan, "i5", "i5", "i3"]}
dict2 = {"ID":[1, 2, 3, 4, 5, 6, 7, 8, 9], "System":[np.nan, np.nan, "win7", np.nan, "win8", "win7", np.nan, np.nan, np.nan],
        "cpu":[np.nan, np.nan, "i3", np.nan, "i7", "i5", np.nan, np.nan, np.nan]}

# 转换两个字典为 DataFrame

df5 = pd.DataFrame(dict1)
df6 = pd.DataFrame(dict2)
print("经过重叠合并后的数据为：\n", df5.combine_first(df6))


# # 将多张菜品订单详情表纵向合并

# In[36]:


import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# 创建数据库连接

conn = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")

# 读取数据

detail1 = pd.read_sql("meal_order_detail1", conn)
detail2 = pd.read_sql("meal_order_detail2", conn)
detail3 = pd.read_sql("meal_order_detail3", conn)

#纵向堆叠 3 张表

detail = detail1.append(detail2)
detail = detail.append(detail3)
print("3张订单详情表合并后的形状为：", detail.shape)


# # 订单详情表、订单信息表、客户信息表主键合并

# In[40]:


order = pd.read_csv("../data/meal_order_info.csv", sep = ",", encoding = "gb18030") # 读取订单信息表
user = pd.read_excel("../data/users_info.xlsx") # 读取客户信息表

# 数据类型转换，存储部分数据

order["info_id"] = order["info_id"].astype("str")
order["emp_id"] = order["emp_id"].astype("str")
user["USER_ID"] = user["USER_ID"].astype("str")
data = pd.merge(detail, order, left_on = ["order_id","emp_id"], right_on = ["info_id", "emp_id"])
data = pd.merge(detail, user, left_on = "emp_id", right_on = "USER_ID", how = "inner")
print("3 张表数据主键合并后的大小为：", data.shape)


