
# coding: utf-8

# # pylot 中的基础绘图语法

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline 表示在行中显示图片，在命令行运行报错

data = np.arange(0, 1.1, 0.01)
plt.title("lines") # 添加标题
plt.xlabel("x") # 添加 x 轴的名称
plt.ylabel("y") # 添加 y 轴的名称
plt.xlim((0, 1)) # 确定 x 轴范围
plt.ylim((0, 1)) # 确定 y 轴范围
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # 规定 x 轴刻度
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # 确定 y 轴刻度
plt.plot(data, data ** 2) # 添加 y = x ^ 2 曲线
plt.plot(data, data ** 4) # 添加 y = x ^ 4 曲线
plt.legend(["y = x ^ 2","y = x ^ 4"])
plt.savefig("../tmp/y = x ^ 2.png")
plt.show()


# # 包含子图绘制的基础语法

# In[3]:


rad = np.arange(0, np.pi * 2, 0.01)

# 第一幅子图

p1 = plt.figure(figsize = (8, 6), dpi = 80) # 确定画布大小
ax1 = p1.add_subplot(2, 1, 1) # 创建一个 2 行 1 列的子图，并开始绘制第一幅
plt.title("lines") # 添加标题
plt.xlabel("x") # 添加 x 轴的名称
plt.ylabel("y") # 添加 y 轴的名称
plt.xlim((0, 1)) # 确定 x 轴范围
plt.ylim((0, 1)) # 确定 y 轴范围
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # 规定 x 轴刻度
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # 确定 y 轴刻度
plt.plot(rad, rad ** 2) # 添加 y = x ^ 2 曲线
plt.plot(rad, rad ** 4) # 添加 y = x ^ 4 曲线
plt.legend(["y = x ^ 2","y = x ^ 4"])

# 第二幅子图

ax2 = p1.add_subplot(2, 1, 2) # 开始绘制第二幅
plt.title("sin/cos") # 添加标题
plt.xlabel("rad") # 添加 x 轴的名称
plt.ylabel("value") # 添加 y 轴的名称
plt.xlim((0, np.pi * 2)) # 确定 x 轴范围
plt.ylim((-1, 1)) # 确定 y 轴范围
plt.xticks([0, np.pi / 2, np.pi, np.pi * 1.5, np.pi * 2]) # 规定 x 轴刻度
plt.yticks([-1, -0.5, 0, 0.5, 1]) # 确定 y 轴刻度
plt.plot(rad, np.sin(rad)) # 添加 sin 曲线
plt.plot(rad, np.cos(rad)) # 添加 cos 曲线
plt.legend(["sin", "cos"])
plt.savefig("../tmp/sincos.png")
plt.show()


# # 调节线条的 rc 参数

# In[4]:


# 原图

x = np.linspace(0, 4 * np.pi) # 生成 x 轴数据
y = np.sin(x) # 生成 y 轴数据
plt.plot(x, y, label = "$sin(x)$") # 绘制 sin 曲线图
plt.title("sin")
plt.savefig("../tmp/默认 sin 曲线.png")
plt.show()


# In[5]:


# 修改后 rc 参数的图

plt.rcParams["lines.linestyle"] = "-."
plt.rcParams["lines.linewidth"] = 3
plt.plot(x, y, label = "$sin(x)$") # 绘制三角函数
plt.title("sin")
plt.savefig("../tmp/修改 rc 参数后 sin 曲线.png")
plt.show()


# # 调节字体的 rc 参数

# In[6]:


# 无法显示中文标题

plt.plot(x, y, label = "$sin(x)$") # 绘制三角函数
plt.title("sin 曲线")
plt.savefig("../tmp/无法显示中文标题 sin 曲线.png")
plt.show()


# In[7]:


# 设置 rc 参数显示中文标题
# 设置字体为 SimHei 显示中文

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False # 设置正常显示符号
plt.plot(x, y, label = "$sin(x)$") # 绘制三角函数
plt.title("sin 曲线")
plt.savefig("../tmp/显示中文标题 sin 曲线.png")
plt.show()


# # 绘制 2000~2017 年各季度国民生产总值散点图

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "SimHei" # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False
data = np.load("../data/国民经济核算季度数据.npz")
name = data["columns"] # 提取其中的 columns 数组，视为数据的标签
values = data["values"] # 提取其中的 values 数组，视为数据的存在位置
plt.figure(figsize = (8, 7)) # 设置画布
plt.scatter(values[:, 0], values[:, 2], marker = "o") # 绘制散点图
plt.xlabel("年份")
plt.ylabel("生产总值（亿元）")
plt.ylim((0, 225000)) # 设置 y 轴范围
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation = 45)
plt.title("2000~2017 年各季度国民生产总值散点图") # 添加图表标题
plt.savefig("../tmp/2000~2017 年各季度国民生产总值散点图.png")
plt.show()


# # 绘制 2000~2017 年间各产业各季度国民生产总值的散点图

# In[10]:


plt.figure(figsize = (8, 7)) # 设置画布

# 绘制散点图 1

plt.scatter(values[:, 0], values[:, 3], marker = "o", c = "red")

# 绘制散点图 2

plt.scatter(values[:, 0], values[:, 4], marker = "D", c = "blue")

# 绘制散点图 3

plt.scatter(values[:, 0], values[:, 5], marker = "v", c = "yellow")
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加纵轴标签
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation = 45)
plt.title("2000~2017 年各产业各季度国民生产总值散点图") # 添加图表标题
plt.legend(["第一产业", "第二产业", "第三产业"]) # 添加图例
plt.savefig("../tmp/2000~2017 年各产业季度国民生产总值散点图.png")
plt.show()


# # 绘制 2000~2017 年各季度国民生产总值折线图

# In[11]:


plt.figure(figsize = (8, 7)) # 设置画布
plt.plot(values[:, 0], values[:, 2], color = "r", linestyle = "--")
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.ylim((0, 225000))
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation = 45)
plt.title("2000~2017 年各季度国民生产总值折线图") # 添加图表标题
plt.savefig("../tmp/2000~2017 年各季度国民生产总值折线图.png")
plt.show()


# # 绘制点线图

# In[12]:


plt.figure(figsize = (8, 7)) # 设置画布
plt.plot(values[:, 0], values[:, 2], color = "r", linestyle = "--", marker = "o") # 绘制点线图
plt.xlabel("年份")
plt.ylabel("生产总值（亿元）")
plt.ylim((0, 225000))
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation = 45)
plt.title("2000~2017 年各季度国民生产总值点线图") # 添加图表标题
plt.savefig("../tmp/2000~2017 年各季度国民生产总值点线图.png")
plt.show()


# # 绘制 2000~2017 年各产业各季度生产总值的折线散点图

# In[3]:


plt.figure(figsize = (8, 7)) # 设置画布
plt.plot(values[:, 0], values[:, 3], "bs-",
        values[:, 0], values[:, 4], "ro-.",
        values[:, 0], values[:, 5], "gH--") # 绘制折线图
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.xticks(range(0, 70, 4),values[range(0, 70, 4), 1], rotation = 45)
plt.title("2000~2017 年各产业各季度国民生产总值折线图") # 添加图表标题
plt.legend(["第一产业", "第二产业", "第三产业"])
plt.savefig("../tmp/2000~2017 年各产业各季度国民生产总值折线散点图.png")
plt.show()


# # 绘制 2000~2017 年各产业与行业的国民生产总值散点图

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "SimHei" # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False
data = np.load("../data/国民经济核算季度数据.npz")
name = data["columns"] # 提取其中的 columns 数组，视为数据的标签
values = data["values"] # 提取其中的 values 数组，视为数据的存在位置
p = plt.figure(figsize = (12, 12)) # 设置画布

# 子图1

ax1 = p.add_subplot(2, 1, 1)
plt.scatter(values[:, 0], values[:, 3], marker = "o", c = "r") # 绘制散点图
plt.scatter(values[:, 0], values[:, 4], marker = "D", c = "b") # 绘制散点图
plt.scatter(values[:, 0], values[:, 5], marker = "v", c = "y") # 绘制散点图
plt.ylabel("生产总值（亿元）") # 添加纵轴标签
plt.title("2000~2017 年各产业与行业各季度国民生产总值散点图") # 添加图表标题
plt.legend(["第一产业", "第二产业", "第三产业"]) # 添加图例

# 子图2

ax2 = p.add_subplot(2, 1, 2)
plt.scatter(values[:, 0], values[:, 6], marker = "o", c = "r") # 绘制散点图
plt.scatter(values[:, 0], values[:, 7], marker = "D", c = "b") # 绘制散点图
plt.scatter(values[:, 0], values[:, 8], marker = "v", c = "y") # 绘制散点图
plt.scatter(values[:, 0], values[:, 9], marker = "8", c = "g") # 绘制散点图
plt.scatter(values[:, 0], values[:, 10], marker = "p", c = "c") # 绘制散点图
plt.scatter(values[:, 0], values[:, 11], marker = "+", c = "m") # 绘制散点图
plt.scatter(values[:, 0], values[:, 12], marker = "s", c = "k") # 绘制散点图

# 绘制散点图

plt.scatter(values[:, 0], values[:, 13], marker = "*", c = "purple")

# 绘制散点图

plt.scatter(values[:, 0], values[:, 14], marker = "d", c = "brown")
plt.legend(["农业", "工业", "建筑", "批发", "交通", "餐饮", "金融", "房地产", "其他"])
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加纵轴标签
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation = 45)
plt.savefig("../tmp/2000~2017 年各产业与行业各季度国民生产总值散点图.png")
plt.show()


# # 绘制 2000~2017 年各产业与行业的国民生产总值折线图

# In[6]:


p1 = plt.figure(figsize = (8, 7)) # 设置画布

# 子图1

ax3 = p1.add_subplot(2, 1, 1)
plt.plot(values[:, 0], values[:, 3], "b-",
        values[:, 0], values[:, 4], "r-.",
        values[:, 0], values[:, 5], "g--") # 绘制折线图
plt.ylabel("生产总值（亿元）") # 添加纵轴标签
plt.title("2000~2017 各产业与行业各季度国民生产总值折线图") # 添加图表标题
plt.legend(["第一产业", "第二产业", "第三产业"]) # 添加图例

# 子图2

ax4=p1.add_subplot(2, 1, 2)
plt.plot(values[:, 0], values[:, 6], "r-", # 绘制折线图
        values[:, 0], values[:, 7], "b-.", # 绘制折线图
        values[:, 0], values[:, 8], "y--", # 绘制折线图
        values[:, 0], values[:, 9], "g:", # 绘制折线图
        values[:, 0], values[:, 10], "c-", # 绘制折线图
        values[:, 0], values[:, 11], "m-.", # 绘制折线图
        values[:, 0], values[:, 12], "k--", # 绘制折线图
        values[:, 0], values[:, 13], "r:", # 绘制折线图
        values[:, 0], values[:, 14], "b-") # 绘制折线图
plt.legend(["农业", "工业", "建筑", "批发", "交通", "餐饮", "金融", "房地产", "其他"])
plt.xlabel("年份") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加纵轴标签
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation = 45)
plt.savefig("../tmp/2000~2017 年各产业与行业各季度国民生产总值折线子图.png")
plt.show()


# # 绘制 2017 年第一季度各产业国民生产总值直方图

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "SimHei" # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False
data = np.load("../data/国民经济核算季度数据.npz")
name = data["columns"] # 提取其中的 columns 数组，视为数据的标签
values = data["values"] # 提取其中的 values 数组，视为数据的存在位置
label = ["第一产业", "第二产业", "第三产业"] # 刻度标签
plt.figure(figsize = (6, 5)) # 设置画布
plt.bar(range(3), values[-1, 3:6], width = 0.5) # 绘制直方图
plt.xlabel("产业") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.xticks(range(3), label)
plt.title("2017 年第一季度各产业国民生产总值直方图") # 添加图表标题
plt.savefig("../tmp/2017 年第一季度各产业国民生产总值直方图.png")
plt.show()


# # 绘制 2017 年第一季度各产业国民生产总值饼图

# In[9]:


plt.figure(figsize = (6, 6)) # 将画布设定为正方形，则绘制的饼图是正圆
label = ["第一产业", "第二产业", "第三产业"] # 定义饼图的标签，标签是列表
explode = [0.01, 0.01, 0.01] # 设定各项距离圆心n个半径
plt.pie(values[-1, 3:6], explode = explode, labels = label, autopct = "%1.1f%%") # 绘制饼图
plt.title("2017 年第一季度各产业国民生产总值饼图")
plt.savefig("../tmp/2017 年第一季度各产业生产总值占比饼图")
plt.show()


# # 绘制 2000~2017 年各产业国民生产总值箱线图

# In[10]:


label = ["第一产业", "第二产业", "第三产业"] # 定义标签
gdp = (list(values[:, 3]), list(values[:, 4]), list(values[:, 5]))
plt.figure(figsize = (6, 4))
plt.boxplot(gdp, notch = True, labels = label, meanline = True)
plt.title("2000~2017 年各产业国民生产总值箱线图")
plt.savefig("../tmp/2000~2017 年各产业国民生产总值箱线图.png")
plt.show()


# # 绘制国民生产总值构成分布直方图

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

data = np.load("../data/国民经济核算季度数据.npz")
name = data["columns"] # 提取其中的 columns 数组，视为数据的标签
values = data["values"] # 提取其中的 values 数组，视为数据的存在位置
plt.rcParams["font.sans-serif"] = "SimHei" # 设置中文显示
plt.rcParams["axes.unicode_minus"] = False
label1 = ["第一产业", "第二产业", "第三产业"] # 刻度标签 1
label2 = ["农业", "工业", "建筑", "批发", "交通", "餐饮", "金融", "房地产", "其他"] # 刻度标签 2
p = plt.figure(figsize = (12, 12))

# 子图 1

ax1 = p.add_subplot(2, 2, 1)
plt.bar(range(3), values[0, 3:6], width = 0.5) # 绘制直方图
plt.xlabel("产业") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.xticks(range(3), label1)
plt.title("2000 年第一季度国民生产总值产业构成分布直方图")

# 子图 2

ax2 = p.add_subplot(2, 2, 2)
plt.bar(range(3), values[-1, 3:6], width = 0.5) # 绘制直方图
plt.xlabel("产业") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.xticks(range(3), label1)
plt.title("2017 年第一季度国民生产总值产业构成分布直方图")

# 子图 3

ax3 = p.add_subplot(2, 2, 3)
plt.bar(range(9), values[0, 6:], width = 0.5) # 绘制直方图
plt.xlabel("行业") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.xticks(range(9), label2)
plt.title("2000 年第一季度国民生产总值行业构成分布直方图") # 添加图表标题

# 子图 4

ax4 = p.add_subplot(2, 2, 4)
plt.bar(range(9), values[-1, 6:], width = 0.5) # 绘制直方图
plt.xlabel("行业") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称
plt.xticks(range(9), label2)
plt.title("2017 年第一季度国民生产总值行业构成分布直方图") # 添加图表标题

# 保存并显示图形

plt.savefig("../tmp/国民生产总值构成分布直方图.png")
plt.show()


# # 绘制国民生产总值构成分布饼图

# In[14]:


label1 = ["第一产业", "第二产业", "第三产业"] # 标签1
label2 = ["农业", "工业", "建筑", "批发", "交通", "餐饮", "金融", "房地产", "其他"] # 标签2
explode1 = [0.01, 0.01, 0.01]
explode2 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
p = plt.figure(figsize = (12, 12))

# 子图 1

ax1 = p.add_subplot(2, 2, 1)
plt.pie(values[0, 3:6], explode = explode1, labels = label1, autopct = "%1.1f%%") # 绘制饼图
plt.title("2000 年第一季度国民生产总值产业构成分布饼图")

# 子图2

ax2 = p.add_subplot(2, 2, 2)
plt.pie(values[-1, 3:6], explode = explode1, labels = label1, autopct = "%1.1f%%") # 绘制饼图
plt.title("2017 年第一季度国民生产总值产业构成分布饼图")

# 子图3

ax3 = p.add_subplot(2, 2, 3)
plt.pie(values[0, 6:], explode = explode2, labels = label2, autopct = "%1.1f%%") # 绘制饼图
plt.title("2000 年第一季度国民生产总值行业构成分布饼图") # 添加图表标题

# 子图4

ax4 = p.add_subplot(2, 2, 4)
plt.pie(values[-1, 6:], explode = explode2, labels = label2, autopct = "%1.1f%%") # 绘制饼图‘
plt.title("2017 年第一季度国民生产总值行业构成分布饼图") # 添加图表标题

# 保存并显示图形

plt.savefig("../tmp/国民生产总值构成分布饼图.png")
plt.show()


# # 绘制国民生产总值分散情况箱线图

# In[3]:


label1 = ["第一产业", "第二产业", "第三产业"] # 标签 1
label2 = ["农业", "工业", "建筑", "批发", "交通", "餐饮", "金融", "房地产", "其他"] # 标签 2
gdp1 = (list(values[:, 3]), list(values[:, 4]), list(values[:, 5]))
gdp2 = ([list(values[:, i]) for i in range(6, 15)])
p = plt.figure(figsize = (8, 8))

# 子图 1

ax1 = p.add_subplot(2, 1, 1)

# 绘制箱线图

plt.boxplot(gdp1, notch = True, labels = label1, meanline = True)
plt.title("2000~2017 年各产业国民生产总值箱线图")
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称

# 子图 2

ax2 = p.add_subplot(2, 1, 2)

# 绘制箱线图

plt.boxplot(gdp2, notch = True, labels = label2, meanline = True)
plt.title("2000~2017 年各行业国民生产总值箱线图")
plt.xlabel("行业") # 添加横轴标签
plt.ylabel("生产总值（亿元）") # 添加 y 轴名称

# 保存并显示图形

plt.savefig("../tmp/国民生产总值分散情况箱线图.png")
plt.show()

