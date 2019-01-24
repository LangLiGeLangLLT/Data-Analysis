
# coding: utf-8

# # SQLAlchemy 连接 MySQL 数据库的代码

# In[1]:


from sqlalchemy import create_engine

# 创建一个 MySQL 连接器，用户为 root，密码为
# 地址为 127.0.0.1，数据库名称为 testdb，编码为 UTF-8

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")
print(engine)


# # 使用 read_sql_table、read_sql_query、read_sql 函数读取数据库数据

# In[2]:


import pandas as pd

# 使用 read_sql_query 查看 testdb 中的数据表数目

formlist = pd.read_sql_query("show tables", con = engine)
print("testdb 数据库数据表清单为：\n", formlist)


# In[3]:


# 使用 read_sql_table 读取订单详情表

detail1 = pd.read_sql_table("meal_order_detail1", con = engine)
print("使用read_sql_table读取订单详情表的长度为：", len(detail1))


# In[4]:


# 使用 read_sql 读取订单详情表

detail2 = pd.read_sql("select * from meal_order_detail2", con = engine)
print("使用 read_sql 函数 + SQL 语句读取的订单详情表长度为：", len(detail2))
detail3 = pd.read_sql("meal_order_detail3", con = engine)
print("使用 read_sql 函数 + 表格名称读取的订单详情表的长度为：", len(detail3))


# # 使用 to_sql 方法写入数据

# In[5]:


# 使用 to_sql 存储 orderData

detail1.to_sql("test1", con = engine, index = False, if_exists = "replace")

# 使用 read_sql 读取 test 表

formlist1 = pd.read_sql_query("show tables", con = engine)
print("新增一个表格后，testdb 数据库数据表清单为：\n", formlist1)


# # 使用 read_table 和 read_csv 函数读取菜品订单信息表

# In[6]:


# 使用 read_table 读取菜品订单信息表

order = pd.read_table("../data/meal_order_info.csv", sep = ",", encoding = "gbk")
print("使用read_table读取的菜品订单信息表的长度为：", len(order))


# In[7]:


# 使用 read_csv 读取菜品订单信息表

order1 = pd.read_csv("../data/meal_order_info.csv", encoding = "gbk")
print("使用read_csv读取的菜品订单信息表的长度为：", len(order1))


# # 更改参数读取菜品订单信息表

# In[8]:


# 使用 read_table 读取菜品订单信息表，sep = ";"

order2 = pd.read_table("../data/meal_order_info.csv", sep = ";",encoding = "gbk")
print("分隔符为 ; 时菜品订单信息为：\n", order2)


# In[9]:


# 使用 read_csv 读取菜品订单信息表，header = None

order3 = pd.read_csv("../data/meal_order_info.csv", sep = ",", header = None, encoding = "gbk")
print("header 为 None 时菜品订单信息表为：\n", order3)


# In[10]:


# 使用 UTF-8 解析菜品订单信息表

order4 = pd.read_csv("../data/meal_order_info.csv", sep = ",", encoding = "utf-8")


# # 使用 to_csv 函数将数据写入 CSV 文件中

# In[11]:


import os

print("菜品订单信息表写入文本文件前目录内文件列表为：\n", os.listdir("../tmp"))

# 将 order 以 CSV 格式存储

order.to_csv("../tmp/orderInfo.csv", sep = ";", index = False)
print("菜品订单信息表写入文本文件后目录内文件列表为：\n", os.listdir("../tmp"))


# # 使用 read_excel 函数读取菜品订单信息表

# In[12]:


user = pd.read_excel("../data/users.xlsx") # 读取 user.xlsx 文件
print("客户信息表长度为：", len(user))


# # 使用 to_excel 函数将数据存储为 Excel 文件

# In[13]:


print("客户信息表写入 Excel 文件前，目录内文件列表为：\n", os.listdir("../tmp"))
user.to_excel("../tmp/userInfo.xlsx")
print("客户信息表写入 Excel 文件后，目录内文件列表为：\n", os.listdir("../tmp"))


# # 读取订单详情表

# In[14]:


# 导入 SQLAlchemy 库的 creat_engine 函数

from sqlalchemy import create_engine
import pandas as pd

# 创建一个 MySQL 连接器，用户名为 root，密码为
# 地址为 127.0.0.1，数据库名称为 testdb

engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/testdb?charset=utf8")

# 使用 read_sql_table 读取订单详情表

order1 = pd.read_sql_table("meal_order_detail1", con = engine)
print("订单详情表 1 的长度为：", len(order1))
order2 = pd.read_sql_table("meal_order_detail2", con = engine)
print("订单详情表 2 的长度为：", len(order2))
order3 = pd.read_sql_table("meal_order_detail3", con = engine)
print("订单详情表 3 的长度为：", len(order3))


# # 读取订单信息表

# In[15]:


# 使用 read_table 读取订单信息表

orderInfo = pd.read_table("../data/meal_order_info.csv", sep = ",", encoding = "gbk")
print("订单信息表的长度为：", len(orderInfo))


# # 读取客户信息表

# In[17]:


# 读取 user.xlsx 文件

userInfo = pd.read_excel("../data/users.xlsx", sheet_name = "users1")
print("客户信息表的长度为：", len(userInfo))

