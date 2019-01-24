
# coding: utf-8

# # 数组的四则运算

# In[1]:

import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("数组相加结果为：", x + y) # 数组相加


# In[2]:

print("数组相减结果为：", x - y) # 数组相减


# In[3]:

print("数组相乘结果为：", x * y) # 数组相乘


# In[4]:

print("数组相除结果为：", x / y) # 数组相除


# In[5]:

print("数组幂运算结果为：", x ** y) # 数组幂运算


# # 数组的比较运算

# In[6]:

x = np.array([1, 3, 5])
y = np.array([2, 3, 4])
print("数组比较结果为：", x < y)


# In[7]:

print("数组比较结果为：", x == y)


# In[8]:

print("数组比较结果为：", x >= y)


# In[9]:

print("数组比较结果为：", x <= y)


# In[10]:

print("数组比较结果为：", x != y)


# # 数组的逻辑运算

# In[11]:

print("数组逻辑运算结果为：", np.all(x == y)) # np.all() 表示逻辑 and


# In[12]:

print("数组逻辑运算结果为：", np.any(x == y)) # np.any() 表示逻辑 or


# # 一维数组的广播机制

# In[14]:

arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
print("创建的数组 1 为：", arr1)


# In[15]:

print("数组 1 的 shape 为：", arr1.shape)


# In[16]:

arr2 = np.array([1, 2, 3])
print("数组 2 的 shape 为：", arr2.shape)


# In[17]:

print("数组相加结果为：", arr1 + arr2)


# # 二维数组的广播机制

# In[18]:

arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
print("创建的数组 1 为：", arr1)


# In[19]:

print("数组 1 的 shape 为：", arr1.shape)


# In[21]:

arr2 = np.array([1, 2, 3, 4]).reshape((4, 1))
print("创建的数组 2 为：", arr2)


# In[22]:

print("数组 2 的 shape 为：", arr2.shape)


# In[23]:

print("数组相加结果为：", arr1 + arr2)


# # 二进制数据存储

# In[9]:

import numpy as np

arr = np.arange(100).reshape(10, 10) # 创建一个数组
np.save("../tmp/save_arr", arr) # 保存数组
print("保存的数组为：\n", arr)


# # 多个数组存储
#
# 如果将多个数组保存到一个文件中，可以使用 savez 函数，其文件的扩展名为 .npz。

# In[11]:

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.arange(0, 1.0, 0.1)
np.savez("../tmp/savez_arr", arr1, arr2)
print("保存的数组 1 为：", arr1)


# In[12]:

print("保存的数组 2 为：", arr2)


# # 二进制文件读取
#
# 需要读取二进制文件的时候可以使用 load 函数，用文件名作为参数。

# In[13]:

loaded_data = np.load("../tmp/save_arr.npy") # 读取含有单个数组的文件
print("读取的数组为：\n", loaded_data)


# In[14]:


# 读取含有多个数组的文件
loaded_data1 = np.load("../tmp/savez_arr.npz")
print("读取的数组 1 为：", loaded_data1['arr_0'])


# # 文件存储与读取
#
# 需要注意的是，存储时可以省略扩展名，但读取时不能省略扩展名。
#
# 在实际的数据分析任务中，更多地使用文本格式的数据，如 TXT 或 CSV 格式，
#
# 因此经常使用 savetxt 函数、loadtxt 函数、genfromtxt 函数执行对文本格式数据的读取任务。

# In[16]:

arr = np.arange(0, 12, 0.5).reshape(4, -1)
print("创建的数组为：", arr)


# In[17]:


# fmt = "%d" 表示保存为整数
np.savetxt("../tmp/arr.txt", arr, fmt = "%d", delimiter = ",")

# 读入的时候也需要指定逗号分隔
loaded_data = np.loadtxt("../tmp/arr.txt", delimiter = ",")
print("读取的数组为：", loaded_data)


# # 使用 genfromtxt 函数读取数组
#
# genfromtxt 函数和 loadtxt 函数相似，不过它面向的是结构化数组和缺失数据。
#
# 它通常使用的参数有 3 个，即存放数据的文件名参数 “fname”、用于分隔的字符参数 “delimiter” 和是否含有列标题参数 “names”。

# In[20]:

loaded_data = np.genfromtxt("../tmp/arr.txt", delimiter = ",")
print("读取的数组为：", loaded_data)


# # 使用 sort 函数进行排序

# In[21]:

np.random.seed(42)
arr = np.random.randint(1, 10, size = 10) # 生成随机数组
print("创建的数组为：", arr)


# In[22]:

arr.sort() # 直接排序
print("排序后数组为：", arr)


# In[23]:

arr = np.random.randint(1, 10, size = (3,3)) # 生成 3 行 3 列的随机数组
print("创建的数组为：", arr)


# In[25]:

arr.sort(axis = 1) # 沿着横轴排序
print("排序后数组为：", arr)


# In[26]:

arr.sort(axis = 0) # 沿着纵轴排序
print("排序后数组为：", arr)


# # 使用 argsort 函数进行排序

# In[27]:

arr = np.array([2, 3, 6, 8, 0, 7])
print("创建的数组为：", arr)


# In[28]:

print("排序后数组为：", arr.argsort()) # 返回值为重新排序值的下标


# # 使用 lexsort 函数进行排序

# In[29]:

a = np.array([3, 2, 6, 4, 5])
b = np.array([50, 30, 40, 20, 10])
c = np.array([400, 300, 600, 100, 200])
d = np.lexsort((a, b, c)) # lexsort 函数只接受一个参数，即 (a, b, c)

# 多个键值排序时是按照最后一个传入数据计算的
print("排序后数组为：",list(zip(a[d], b[d], c[d])))


# # 数组内数据去重

# In[6]:

names = np.array(["小明", "小黄", "小花", "小明", "小花", "小兰", "小白"])
print("创建的数组为：", names)


# In[7]:

print("去重后的数组为：", np.unique(names))


# In[9]:


# 跟 np.unique 等价的 Python 代码实现过程
print("去重后的数组为：", sorted(set(names)))


# In[10]:

ints = np.array([1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10])
print("创建的数组为：", ints)


# In[11]:

print("去重后的数组为：", np.unique(ints))


# # 使用 tile 函数实现数据重复

# In[12]:

arr = np.arange(5)
print("创建的数组为：", arr)


# In[14]:

print("重复后数组为：", np.tile(arr, 3)) # 对数组进行重复


# # 使用 repeat 函数实现数据重复

# In[15]:

np.random.seed(42)
arr = np.random.randint(0, 10, size = (3, 3))
print("创建的数组为：", arr)


# In[17]:

print("重复后数组为：", arr.repeat(2, axis = 0)) # 按行进行元素重复


# In[18]:

print("重复后数组为：", arr.repeat(2, axis = 1)) # 按列进行元素重复


# # NumPy 中常用统计函数的使用

# In[19]:

arr = np.arange(20).reshape(4, 5)
print("创建的数组为：", arr)


# In[20]:

print("数组的和为：", np.sum(arr)) # 计算数组的和


# In[21]:

print("数组纵轴的和为：", arr.sum(axis = 0)) # 沿着纵轴求和


# In[22]:

print("数组横轴的和为：", arr.sum(axis = 1)) # 沿着横轴求和


# In[23]:

print("数组的均值为：", np.mean(arr)) # 计算数组均值


# In[24]:

print("数组纵轴的均值为：", arr.mean(axis = 0)) # 沿着纵轴计算数组均值


# In[25]:

print("数组横轴的均值为：", arr.mean(axis = 1)) # 沿着横轴计算数组均值


# In[26]:

print("数组的标准差为：", np.std(arr)) # 计算数组标准差


# In[27]:

print("数组的方差为：", np.var(arr)) # 计算数组方差


# In[28]:

print("数组的最小值为：", np.min(arr)) # 计算数组最小值


# In[29]:

print("数组的最大值为：", np.max(arr)) # 计算数组最大值


# In[30]:

print("数组的最小元素索引为：", np.argmin(arr)) # 返回数组最小元素的索引


# In[31]:

print("数组的最大元素索引为：", np.argmax(arr)) # 返回数组最大元素的索引


# # cunsum 函数和 cumprod 函数的使用

# In[32]:

arr = np.arange(2, 10)
print("创建的数组为：", arr)


# In[33]:

print("数组元素的累计和为：", np.cumsum(arr)) # 计算所有元素的累计和


# In[34]:

print("数组元素的累计积威：", np.cumprod(arr)) # 计算所有元素的累计积


# # 花萼长度数据统计分析

# In[4]:

iris_sepal_length = np.loadtxt("../data/iris_sepal_length.csv", delimiter = ",") # 读取文件
print("花萼长度表示为：", iris_sepal_length)


# In[5]:

iris_sepal_length.sort() # 对数据进行排序
print("排序后的花萼长度表为：", iris_sepal_length)


# In[6]:


# 去除重复值
print("去重后的花萼长度表为：", np.unique(iris_sepal_length))


# In[8]:

print("花萼长度表的总和为：", np.sum(iris_sepal_length)) # 计算数组总和


# In[9]:


# 计算所有元素的累计和
print("花萼长度表的累计总和为：", np.cumsum(iris_sepal_length))


# In[10]:

print("花萼长度表的均值为：", np.mean(iris_sepal_length)) # 计算数组均值


# In[11]:


# 计算数组标准差
print("花萼长度表的标准差为：", np.std(iris_sepal_length))


# In[12]:

print("花萼长度表的方差为：", np.var(iris_sepal_length)) # 计算数组方差


# In[13]:

print("花萼长度表的最小值为：", np.min(iris_sepal_length)) # 计算最小值


# In[14]:

print("花萼长度表的最大值为：", np.max(iris_sepal_length)) # 计算最大值

