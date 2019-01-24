
# coding: utf-8

# # 实训 1 分析 1996~2015 年人口数据特征间的关系

# In[32]:


import numpy as np
import matplotlib.pyplot as plt

population_data = np.load("../data/populations.npz") # 读取数据
population_data.files # 查看索引


# In[33]:


name = population_data["feature_names"] # 提取标签
values = population_data["data"] # 提取数据
print(name)
print(values)
values = values[0:20] # 切分 values
print(values)


# In[35]:


plt.rcParams["font.sans-serif"] = "SimHei" # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False
p = plt.figure(figsize = (12, 12)) # 设置画布

# 散点图

ax1 = p.add_subplot(2, 1, 1)
plt.scatter(values[::-1, 0], values[::-1, 1], marker = "o", c = "r") # 年末总人口散点图
plt.scatter(values[::-1, 0], values[::-1, 2], marker = "D", c = "b") # 男性人口散点图
plt.scatter(values[::-1, 0], values[::-1, 3], marker = "v", c = "y") # 女性人口散点图
plt.scatter(values[::-1, 0], values[::-1, 4], marker = "8", c = "g") # 城镇人口散点图
plt.scatter(values[::-1, 0], values[::-1, 5], marker = "p", c = "c") # 乡村人口散点图
plt.ylabel("人口数量（万人）") # 添加纵轴标签
plt.title("1996~2015 年人口数据散点图") # 添加图表标题
plt.legend(["年末总人口", "男性人口", "女性人口", "城镇人口", "乡村人口"]) # 添加图例
plt.xticks(rotation = 45)

# 折线图

ax2 = p.add_subplot(2, 1, 2)
plt.plot(values[::-1, 0], values[::-1, 1], "r-", # 年末总人口折线图
        values[::-1, 0], values[::-1, 2], "b-.", # 男性人口折线图
        values[::-1, 0], values[::-1, 3], "y--", # 女性人口折线图
        values[::-1, 0], values[::-1, 4], "g:", # 城镇人口折线图
        values[::-1, 0], values[::-1, 5], "c-") # 乡村人口折线图
plt.ylabel("人口数量（万人）") # 添加纵轴标签
plt.title("1996~2015 年人口数据折线图") # 添加图表标题
plt.legend(["年末总人口", "男性人口", "女性人口", "城镇人口", "乡村人口"]) # 添加图例
plt.xticks(rotation = 45)
plt.savefig("../tmp/1996~2015 年人口数据散点图和折线图.png")
plt.show()


# 分析未来人口变化趋势：
# 
# 1.总人口随着年份的增长而增加。2.男性人口随着年份的增长而增加。3.女性人口随着年份的增长而增加。4.城镇人口随着年份的增长而增加。5.乡村人口随着年份的增长而减少。

# # 实训 2 分析 1996~2015 年人口数据各个特征的分布与分散状况

# In[12]:


import numpy as np
import matplotlib.pyplot as plt

population_data = np.load("../data/populations.npz") # 读取数据
name = population_data["feature_names"] # 提取标签
values = population_data["data"][0:20] # 提取数据
plt.rcParams["font.sans-serif"] = "SimHei" # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False

# 男女人口数目直方图

p1 = plt.figure(figsize = (12, 5))
ax1 = p1.add_subplot(1, 2, 1)
plt.bar(np.arange(0, 20) - 0.25, values[::-1, 2], width = 0.5) # 绘制直方图
plt.bar(np.arange(0, 20) + 0.25, values[::-1, 3], width = 0.5)
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("人口数量（万人）") # 添加 y 轴名称
plt.xticks(range(20), values[::-1, 0], rotation = 45)
plt.legend(["男性人口", "女性人口"]) # 添加图例
plt.title("1996~2015 年各年份男女人口数目直方图")

# 城乡人口数目直方图

ax2 = p1.add_subplot(1, 2, 2)
plt.bar(np.arange(0, 20) - 0.25, values[::-1, 4], width = 0.5) # 绘制直方图
plt.bar(np.arange(0, 20) + 0.25, values[::-1, 5], width = 0.5)
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("人口数量（万人）") # 添加 y 轴名称
plt.xticks(range(20), values[::-1, 0], rotation = 45)
plt.legend(["城镇人口", "乡村人口"]) # 添加图例
plt.title("1996~2015 年各年份城乡人口数目直方图")
plt.savefig("../tmp/1996~2015 年各年份城乡人口数目直方图.png")

# 男女人口比例饼图

p2 = plt.figure(figsize = (12, 5))
ax3 = p2.add_subplot(1, 2, 1)
plt.pie(values[0, (2, 3)], explode = [0.01, 0.01], labels = ["男性人口", "女性人口"], autopct = "%1.1f%%") # 绘制饼图
plt.title("1996~2015年男女人口比例饼图")

# 城乡人口比例饼图

ax4 = p2.add_subplot(1, 2, 2)
plt.pie(values[0, (4, 5)], explode = [0.01, 0.01], labels = ["城镇人口", "乡村人口"], autopct = "%1.1f%%")
plt.title("1996~2015 年男女人口比例饼图")
plt.savefig("../tmp/1996~2015 年男女人口比例饼图.png")

# 男女人口变化速率箱线图

p3 = plt.figure(figsize = (12, 5))
ax5 = p3.add_subplot(1, 2, 1)
gdp1 = (list(values[::-1, 2]), list(values[::-1, 3]))
label1 = ["男性人口", "女性人口"]
plt.boxplot(gdp1, notch = True, labels = label1, meanline = True)
plt.ylabel("人口数量（万人）")
plt.title("1996~2015 年男女人口变化速率箱线图")

# 城乡人口变化速率箱线图

ax6 = p3.add_subplot(1, 2, 2)
gdp2 = (list(values[::-1, 4]), list(values[::-1, 5]))
label2 = ["城镇人口", "乡村人口"]
plt.boxplot(gdp2, notch = True, labels = label2, meanline = True)
plt.ylabel("人口数量（万人）")
plt.title("1996~2015 年城乡人口变化速率箱线图")
plt.savefig("../tmp/1996~2015 年城乡人口变化速率箱线图.png")

plt.show()


# 分析我国人口结构变化情况以及变化速率的增减状况：
# 
# 男女人口数目均逐年增长，城镇人口逐年增加，乡村人口逐年减少。
