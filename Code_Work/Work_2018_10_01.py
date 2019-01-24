
# coding: utf-8

# # 1.矩阵相乘

# In[7]:


import numpy as np

matrix1 = np.matrix([[-2, 4], [1, -2]])
matrix2 = np.matrix([[2, 4], [-3, -6]])
print("结果为：\n", matrix1 * matrix2)


# In[13]:


import numpy as np

matrix1 = np.matrix([[1, 2, 3], [1, 0, -1]])
matrix2 = np.matrix([[1, 0, -1, 2], [-1, 1, 3, 0], [0, -2, -1, 3]])
print("结果为：\n", matrix1 * matrix2)


# # 2.解线性方程

# In[14]:


import numpy as np
from numpy.linalg import *

m = np.matrix([[1, 1, 1], [1, 2, 3], [1, 10, 100]])
y = np.matrix([[3], [6], [111]])
print("x 的解集为：\n", solve(m, y))


# # 3.解应用题

# In[15]:


import numpy as np
from numpy.linalg import *

m = np.matrix([[3, 3.2], [3.5, 3.6]])
y = np.matrix([[118.4], [135.2]])
print("x 的解集为：\n", solve(m, y))

