
# coding: utf-8

# # Python 的内建对象数组可以有三种形式：
#
# List 列表：[1, 2, 3]
#
# Tuple 元组：(1, 2, 3, 4, 5)
#
# Dict 字典：{A:1, B:2}
#
# 列表为大家所熟知，元组与列表相似，区别在于元组里的值无法修改。字典是另一种可变容器模型，且可存储任意类型对象。
# 字典对象由键和值组成，每个键值 (key -> value) 对用冒号 (:) 分割，每个对之间用逗号 (,) 分割，整个字典包括在
# 花括号 ({}) 中
#
# Python 中也提供了一个 array 模块，只保存数值，不支持多维运算，也没有各种运算的函数，因此不适合做数组运算。
# NumPy 诞生弥补了这些不足。
#
# numpy 有 2 种基本对象，ndarray（N-dimensional array object）和 ufunc（universal functionobject），
# ndarray 是存储单一数据类型的多维数组， 而 ufunc 则是能够对数组进行处理的函数

# # 通过 list 创建 numpy 的 ndarray 数组

# In[1]:

import numpy as np # 导入 numpy 包，并重名
list_1 = [1, 2, 3, 4]
list_1
print(type(list_1))


# In[3]:

array_1 = np.array(list_1)
array_1
print(type(array_1))


# In[2]:

list_2 = [5, 6, 7, 8]
array_2 = np.array([list_1, list_2]) # 创建二维数组
array_2


# # 数组的属性及说明
#
# 在 NumPy 中维度 (dimensions) 叫做轴 (axes)，轴的个数叫做秩 (rank)。 例如，在 3D 空间一个点的坐标 [1,2,3]
# 是一个秩为 1 的数组，因为它只有一个轴。又例如，在以下例子中，数组的秩为 2 (它有两个维度).

# In[3]:

print('数组轴的个数：', array_2.ndim) # ndarray.ndim：数组轴的个数，在 python 的世界中，轴的个数被称作秩


# In[6]:

print('数组的维度：', array_2.shape) # ndarray.shape：数组的维度。这是一个指示数组在每个维度上大小的整数元组。例如一个 n 排 m 列的矩阵


# In[7]:

print('数组元素的个数：', array_2.size) # ndarray.size：数组元素的总个数，等于 shape 属性中元组元素的乘积。


# In[4]:

print('数组中每个元素的大小：', array_2.itemsize) # ndarray.itemsize：数组中每个元素的字节大小。
                                                # 例如，一个元素类型为 float64 的数组 itemsize 属性值为 8 (= 64 / 8)。


# In[9]:

print('数组类型为：',array_2.dtype) # ndarray.dtype：一个用来描述数组中元素类型的对象，可以通过创造或指定 dtype 使用标准 Python 类型。
                                   # 另外 NumPy 提供它自己的数据类型。


# In[10]:

array_3 = np.array([[1.0, 2, 3], [4.0, 5, 6]])
print('数组类型为：', array_3.dtype)


# In[11]:

print('数组元素的缓冲区', array_2.data) # ndarray.data：包含实际数组元素的缓冲区，通常我们不需要使用这个属性，因为我们总是通过索引来使用数组中的元素。


# # 数组的类型
#
# Numpy 的基本数据类型
# bool_, int_, intc, intp, int8, int16, int32, int64, uint8, unint16, uint32, 
# uint64, float_, float16, float32, float64, complex_, complex64, complex128

# In[5]:

array_4 = np.array(list_1, dtype = np.float) # 也可在定义数组时指定数据类型
array_4


# # 用 arange 创建 ndarray 数组

# In[6]:

array_5 = np.arange(1, 10) # 不包括 10
array_5


# In[7]:

array_6 = np.arange(1, 10, 2) # 不包括 10
array_6


# # 利用 random 快速创建数组
#
# 均匀分布和正态分布的随机函数区别：
# 使用的随机数函数 rand() 产生的是服从均匀分布的随机数，能够模拟等概率出现的情况，例如扔一个骰子，1 到 6 点的概率应该相等。
#
# 但现实生活中更多的随机现象是符合正态分布的，例如20岁成年人的体重分布等。
# 假如我们在制作一个游戏，要随机设定许多人为角色的身高，如果还用均匀分布，生成从 140 到 220 之间的数字，就会发现每个身高段的人数是一样多的，
# 这是比较无趣的，这样的世界也与我们习惯不同，现实应该是特别高和特别矮的都很少，处于中间的人数最多，这就要求随机函数符合正态分布。

# In[16]:

print('均匀分布随机函数 Rand：')
print(np.random.rand()) # 服从均匀分布 0-1 之间随机数
print(np.random.rand(10)) # 服从均匀分布 0-1 之间随机数
print(np.random.rand(2, 4)) # 2 行 4 列的随机数矩阵 ，服从均匀分布 0-1 之间随机数


# In[17]:

print('正态分布随机函数 Randn：')
print(np.random.randn(2, 4)) # 2 行 4 列的随机数矩阵 ，服从正态分布 0-1 之间随机数


# In[78]:

print('RandInt:')
print(np.random.randint(1, 10)) # 生成 1-10 的随机整数


# In[94]:

print('RandInt:')
print(np.random.randint(10, 20, 5)) # 生成 5 个 10-20 的随机整数


# In[93]:

print('Choice:')
print(np.random.choice([10, 20, 30, 2, 8])) # 尽在给出的可选值内产生随机数


# # 用 linspace 和 logspace 函数创建数组

# In[22]:

array_7 = np.linspace(1, 10, 5) # linspace 通过指定开始值、终值和元素个数来创建一维数组，默认设置包括终值，等差数列分布
array_7


# In[23]:

array_8 = np.logspace(1, 10, 5) # logspace 和 linspace 函数类似，它是来创建等比数列的。
array_8


# # 创建特殊数组矩阵

# In[24]:

np.zeros(5) # 全 0 矩阵


# In[25]:

np.zeros([2, 3])


# In[26]:

np.eye(5) # 单位矩阵


# In[27]:

np.ones([4, 3]) # 全 1 矩阵


# In[28]:

np.diag([1, 2, 3, 4]) # 对角线数组，即除对角线以外的其他元素都为 0，对角线上的元素可以是 0 或其他值


# # 数组的访问（索引 和 切片）

# In[29]:

a = np.arange(1, 10)
a


# In[30]:

a[1] # 下标从 0 开始


# In[31]:

a[1:5] # 切片不包含下标为 5 的元素


# In[32]:

b = np.array([[1, 2, 3], [4, 5, 6]])
b


# In[33]:

b[1][0]


# In[34]:

b[1, 0]


# In[35]:

c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
c


# In[36]:

c[:2, 1:] # 二维数组切片， 第 0 行到第 2 行 0 1 第 1 列到最后 取值 1 2


# # 数组形态变化

# In[37]:

arr = np.arange(12)
print('创建一维数组为：', arr)


# In[39]:

print('新的数组为：\n', arr.reshape(3, 4)) # 设置数组的形状


# In[40]:

print('新的数组维度为：\n', arr.reshape(3, 4).ndim)


# In[41]:

print(arr)


# In[43]:

arr = arr.reshape(3, 4)
print(arr)


# In[44]:

print('数组展平后为：\n', arr.ravel()) # 数组横向展平


# In[45]:

print('数组展平后为：\n', arr.flatten()) # 数组横向展平


# In[46]:

print('数组展平后为：\n', arr.flatten('F')) # 数组纵向展平


# In[49]:

arr1 = np.arange(12).reshape(3, 4)
print('创建数组1为：\n', arr1)


# In[51]:

arr2 = arr1 * 3
print('创建数组1为：\n', arr2)


# In[52]:

print('横向组合为：\n', np.hstack((arr1, arr2)))


# In[53]:

print('纵向组合为：\n', np.vstack((arr1, arr2)))


# In[99]:

arr = np.arange(16).reshape(4, 4)
print('创建数组1为：\n', arr)


# In[103]:

print('横向分割为：\n', np.hsplit(arr, 2))


# In[56]:

print('纵向分割为：\n', np.vsplit(arr, 2))


# In[57]:

print('纵向分割为：\n', np.split(arr, 2, axis = 0))


# # NumPy 的 ndarray 快速的元素级数组函数

# In[58]:

print(np.arange(1, 11).reshape([2, -1])) # reshape中 -1 为系统自动计算第二维个数


# In[60]:

array = np.arange(1, 11).reshape([2, -1])
print('Exp:')
print(np.exp(array))
print('Exp2:')
print(np.exp2(array))
print('Sqrt:')
print(np.sqrt(array))
print('Sin:')
print(np.sin(array))
print('Log:')
print(np.log(array))


# # 数组的运算

# In[61]:

a = np.random.randint(10, size = 20).reshape(4, 5)
a


# In[62]:

b = np.random.randint(10, size = 20).reshape(4, 5)
b


# In[63]:

a + b


# In[64]:

a - b


# In[65]:

a * b


# In[66]:

a / b


# # 创建NumPy矩阵以及矩阵的运算
#
# 矩阵是继承自NumPy数组对象的二维数组对象，矩阵是narray的子类，使用mat、matrix、bmat函数创建

# In[69]:

matr1 = np.mat('1,2,3;4,5,6;7,8,9')
print('创建矩阵为：\n', matr1)


# In[73]:

matr2 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('创建矩阵为：\n', matr2)


# In[74]:

arr1 = np.eye(3)
print('创建数组1为：\n', arr1)


# In[75]:

arr2 = 3 * arr1
print('创建数组2为：\n', arr2)


# In[76]:

print('创建矩阵为：\n', np.bmat('arr1 arr2;arr1 arr2'))


# In[77]:

matr2 = matr1 * 3
print('创建矩阵为：\n', matr2)


# In[78]:

print('矩阵相加为：\n', matr1 + matr2)


# In[79]:

print('矩阵相减为：\n', matr1 - matr2)


# In[80]:

print('矩阵相乘为：\n', matr1 * matr2)


# In[81]:

print('矩阵相除为：\n', matr1 / matr2)


# In[82]:

print('矩阵对应元素相乘为：\n', np.multiply(matr1, matr2)) # 对应元素相乘


# In[105]:


# 线性代数
# numpy.linalg 模块包含线性代数的函数。使用这个模块，可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等。
from numpy.linalg import *

print(np.eye(3))


# In[106]:

m = np.array([[1., 2.], [-1., -3.]])
print(m)


# In[107]:

print('Inv:')
print(inv(m))


# In[108]:

print('T:')
print(m.transpose()) #矩阵的转置


# In[109]:

print('Det:')
print(det(m))  # 行列式的值 1 * - 3 - (2 * (-1)))


# In[110]:


#解线性方程组
y = np.array([[5.], [7.]])
print('Solve')
print(m)

# x + 2y = 5
# -x - 3y = 7
print(solve(m, y))

