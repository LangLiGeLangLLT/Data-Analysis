
# coding: utf-8

# # 提取用户信息更新表和登录信息表的时间信息

# In[1]:


import pandas as pd

userupdate_data = pd.read_csv("../data/Training_Userupdate.csv", sep = ",", encoding = "gbk")
loginfo_data = pd.read_csv("../data/Training_LogInfo.csv", sep = ",", encoding = "gbk")

# 使用 to_datetime 函数转换用户信息更新表和登录信息表的时间字符串

userupdate_data["ListingInfo1"] = pd.to_datetime(userupdate_data["ListingInfo1"])
userupdate_data["UserupdateInfo2"] = pd.to_datetime(userupdate_data["UserupdateInfo2"])
loginfo_data["Listinginfo1"] = pd.to_datetime(loginfo_data["Listinginfo1"])
loginfo_data["LogInfo3"] = pd.to_datetime(loginfo_data["LogInfo3"])

# 使用 year、month、week 等方法提取用户信息更新表和登录信息表中的时间信息

userupdate_data_ListingInfo1_year = [i.year for i in userupdate_data["ListingInfo1"]]
print("用户信息更新表中的 ListingInfo1 的年份数据前 5 个为：\n", userupdate_data_ListingInfo1_year[:5])
userupdate_data_ListingInfo1_month = [i.month for i in userupdate_data["ListingInfo1"]]
print("用户信息更新表中的 ListingInfo1 的月份数据前 5 个为：\n", userupdate_data_ListingInfo1_month[:5])
userupdate_data_ListingInfo1_weekday = [i.weekday_name for i in userupdate_data["ListingInfo1"]]
print("用户信息更新表中的 ListingInfo1 的星期名称数据前 5 个为：\n", userupdate_data_ListingInfo1_weekday[:5])

userupdate_data_UserupdateInfo2_year = [i.year for i in userupdate_data["UserupdateInfo2"]]
print("用户信息更新表中的 UserupdateInfo2 的年份数据前 5 个为：\n", userupdate_data_UserupdateInfo2_year[:5])
userupdate_data_UserupdateInfo2_month = [i.month for i in userupdate_data["UserupdateInfo2"]]
print("用户信息更新表中的 UserupdateInfo2 的月份数据前 5 个为：\n", userupdate_data_UserupdateInfo2_month[:5])
userupdate_data_UserupdateInfo2_weekday = [i.weekday_name for i in userupdate_data["UserupdateInfo2"]]
print("用户信息更新表中的 UserupdateInfo2 的星期名称数据前 5 个为：\n", userupdate_data_UserupdateInfo2_weekday[:5])

loginfo_data_Listinginfo1_year = [i.year for i in loginfo_data["Listinginfo1"]]
print("登录信息表中的 Listinginfo1 的年份数据前 5 个为：\n", loginfo_data_Listinginfo1_year[:5])
loginfo_data_Listinginfo1_month = [i.month for i in loginfo_data["Listinginfo1"]]
print("登录信息表中的 Listinginfo1 的月份数据前 5 个为：\n", loginfo_data_Listinginfo1_month[:5])
loginfo_data_Listinginfo1_weekday = [i.weekday_name for i in loginfo_data["Listinginfo1"]]
print("登录信息表中的 Listinginfo1 的星期名称数据前 5 个为：\n", loginfo_data_Listinginfo1_weekday[:5])

loginfo_data_LogInfo3_year = [i.year for i in loginfo_data["LogInfo3"]]
print("登录信息表中的 LogInfo3 的年份数据前 5 个为：\n", loginfo_data_LogInfo3_year[:5])
loginfo_data_LogInfo3_month = [i.month for i in loginfo_data["LogInfo3"]]
print("登录信息表中的 LogInfo3 的月份数据前 5 个为：\n", loginfo_data_LogInfo3_month[:5])
loginfo_data_LogInfo3_weekday = [i.weekday_name for i in loginfo_data["LogInfo3"]]
print("登录信息表中的 LogInfo3 的星期名称份数据前 5 个为：\n", loginfo_data_LogInfo3_weekday[:5])

# 计算用户信息更新表和登录信息表中两时间的差，分别以日、小时、分钟计算

userupdate_data_difference = userupdate_data["ListingInfo1"] - userupdate_data["UserupdateInfo2"]
print("用户信息更新表中两时间的差的数据前5个为：\n", userupdate_data_difference[:5])
loginfo_data_difference = loginfo_data["Listinginfo1"] - loginfo_data["LogInfo3"]
print("登录信息表中两时间的差的数据前 5 个为：\n", loginfo_data_difference[:5])


# # 使用分组聚合方法进一步分析用户信息更新表和登录信息表

# In[2]:


import pandas as pd
import numpy as np

userupdate_data = pd.read_csv("../data/Training_Userupdate.csv", sep = ",", encoding = "gbk")
loginfo_data = pd.read_csv("../data/Training_LogInfo.csv", sep = ",", encoding = "gbk")

# 使用 groupby 方法对用户信息更新表和登录信息表进行分组

userupdate_data_group = userupdate_data[["Idx", "ListingInfo1", "UserupdateInfo1", "UserupdateInfo2"]].groupby(by = "Idx")
loginfo_data_group = loginfo_data[["Idx", "Listinginfo1", "LogInfo1", "LogInfo2", "LogInfo3"]].groupby(by = "Idx")

# 使用 agg 方法求取分组后的最早和最晚更新及登录时间

print("用户信息更新表的分组后的最早和最晚更新时间为：\n", userupdate_data_group.agg({"UserupdateInfo2":[np.min, np.max]}))
print("登录信息表的分组后的最早和最晚登录时间为：\n", loginfo_data_group.agg({"LogInfo3":[np.min, np.max]}))

# 使用 size 方法求取分组后的数据的信息更新次数与登录次数

print("用户信息更新表的分组后的数据的信息更新次数为：\n", userupdate_data_group.agg({"UserupdateInfo2":np.size}))
print("登录信息表的分组后的数据的登录次数为：\n", loginfo_data_group.agg({"LogInfo3":np.size}))


# # 对用户信息更新表和登录信息表进行长宽表转换

# In[4]:


import pandas as pd
import numpy as np

userupdate_data = pd.read_csv("../data/Training_Userupdate.csv", sep = ",", encoding = "gbk")
loginfo_data = pd.read_csv("../data/Training_LogInfo.csv", sep = ",", encoding = "gbk")

# 使用 merge 函数将用户信息更新表和登录信息表以 Idx 为主键进行合并

userupdate_loginfo_data = pd.merge(userupdate_data, loginfo_data, left_on = "Idx", right_on = "Idx")

# 使用 pivot_table 函数进行长宽表转换

userupdate_loginfo_data_pivot = pd.pivot_table(userupdate_loginfo_data, index = "Idx", columns = "UserupdateInfo1", values = ["LogInfo1", "LogInfo2"], aggfunc = np.sum, fill_value = 0, margins = True)
print(userupdate_loginfo_data_pivot)

#使用 crosstab 方法进行长宽表转换

userupdate_loginfo_data_pivot_Loginfo1 = pd.crosstab(index = userupdate_loginfo_data["Idx"], columns = userupdate_loginfo_data["UserupdateInfo1"], values = userupdate_loginfo_data["LogInfo1"], aggfunc = np.sum, margins = True)
print(userupdate_loginfo_data_pivot_Loginfo1)
userupdate_loginfo_data_pivot_Loginfo2 = pd.crosstab(index = userupdate_loginfo_data["Idx"], columns = userupdate_loginfo_data["UserupdateInfo1"], values = userupdate_loginfo_data["LogInfo2"], aggfunc = np.sum, margins = True)
print(userupdate_loginfo_data_pivot_Loginfo2)

