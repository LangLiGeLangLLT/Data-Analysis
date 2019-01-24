
# coding: utf-8

# # 实训 1 创建数组并进行运算

# （1）创建一个数值范围为 0~1，间隔为 0.01 的数组。

# In[3]:


import numpy as np

arr = np.arange(0, 1, 0.01)
print("创建的数组为：\n", arr)


# （2）创建 100 个服从正态分布的随机数。

# In[5]:


import numpy as np

arr = np.random.randn(100)
print("创建的数组为：\n", arr)


# （3）对创建的两个数组进行四则运算。

# In[7]:


import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([6, 7, 8, 9])
print("两个数组相加为：", arr1 + arr2)
print("两个数组相减为：", arr1 - arr2)
print("两个数组相乘为：", arr1 * arr2)
print("两个数组相除为：", arr1 / arr2)
print("两个数组幂运算为：", arr1 ** arr2)


# （4）对创建的随机数组进行简单的统计分析。

# In[15]:


import numpy as np

arr = np.random.randint(10, size = 20).reshape(4, 5)
print("创建的随机数组为：\n", arr)


# In[16]:


print("数组的和为：", np.sum(arr))
print("数组纵轴的和为：", arr.sum(axis = 0))
print("数组横轴的和为：", arr.sum(axis = 1))
print("数组的均值为：", np.mean(arr))
print("数组纵轴的均值为：", arr.mean(axis = 0))
print("数组横轴的均值为：", arr.mean(axis = 1))
print("数组的标准差为：", np.std(arr))
print("数组的方差为：", np.var(arr))
print("数组的最小值为：", np.min(arr))
print("数组的最大值为：", np.max(arr))
print("数组的最小元素索引为：", np.argmin(arr))
print("数组的最大元素索引为：", np.argmax(arr))


# # 实训 2 创建一个国际象棋的棋盘

# （1）创建一个 8 * 8 矩阵。

# In[2]:


import numpy as np

matr = np.matrix(np.zeros([8, 8]))
print(matr)


# （2）把 1、3、5、7 行和 2、4、6 列的元素设置为 1 。

# In[5]:


import numpy as np

arr = np.zeros([8, 8])
i = 0
while(i < 64):
    x = int(i / 8)
    y = int(i % 8)
    if(x %2 == 0 and y % 2 == 0):
        arr[x][y] = 1
    elif(x %2 == 1 and y % 2 == 1):
        arr[x][y] = 1
    i += 1
print(arr)


# In[8]:


import numpy as np

i = 0
k = 1
while(i < 64):
    x = int(i / 8)
    y = int(i % 8)
    if(x % 2 == 0 and y % 2 == 0):
        print("█", end = "")
    elif(x % 2 == 1 and y % 2 == 1):
        print("█", end = "")
    else:
        print("  ", end = "")
    if(k % 8 == 0):
        print()
    i += 1
    k += 1


# # 操作题

# （1）生成范围在 0~1、服从均匀分布的 10 行 5 列的数组。

# In[6]:


import numpy as np

arr = np.random.rand(10, 5)
print(arr)


# （2）生成两个 2 * 2 矩阵，并计算矩阵乘积。

# In[8]:


import numpy as np

matr1 = np.matrix([[1, 2], [3, 4]])
matr2 = np.matrix([[6, 7], [8, 9]])
print("乘积为：\n", matr1 * matr2)


# # 读取 world_alcohol 的文件，统计酒精消费总和以及平均值
# 
# 提示 Display Value 有些字段是没有值的，要将空值 nan 进行预处理，可用 0 替换，
# 
# 数据文件在第二章资料里面，包括 iris 鸢尾花数据和 world_alcohol 饮酒数据。

# In[5]:


import numpy as np

world_alcohol = np.genfromtxt("../data/world_alcohol.txt", delimiter = ",", skip_header = 1)
print("以二进制方式读取数据为：\n", world_alcohol)

is_value_empty = np.isnan(world_alcohol[:, 4])
print("判断哪些数据为 NULL：\n", is_value_empty)

world_alcohol[is_value_empty, 4] = '0' # 有些字段是没有值的，要将空值 nan 进行预处理，可用0替换
alcohol_consumption = world_alcohol[:, 4]
alcohol_consumption = alcohol_consumption.astype(float)
total_alcohol = alcohol_consumption.sum()
average_alcohol = alcohol_consumption.mean()
print("酒精消费总和为：", total_alcohol)
print("酒精消费平均值为：", average_alcohol)

