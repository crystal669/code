#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os


# In[10]:


read_path1='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\train_log.csv'
read_path2='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\test_log.csv'

save_path='F:\DeepLearning\DataSet\mydataset\process\kddcup\\kddData_info.csv'


# In[ ]:


train_data=pd.read_csv(read_path1)
test_data=pd.read_csv(read_path2)


# In[ ]:


data=pd.concat((train_data,test_data),axis=0)


# In[ ]:


groups=data.groupby(data['enroll_id'])
list_enroll=data['enroll_id'].drop_duplicates().dropna().to_list()
Data=pd.DataFrame()
en_data=pd.DataFrame()

for en in list_enroll:
    en_data=groups.get_group(en)
    #重置索引必须inplace=True,drop=True去掉index列
    en_data.reset_index(drop=True,inplace=True)
    #按照日期排序
    #print("sort:")
    en_data=en_data.sort_values(by='time',ascending=True)
    if(list_enroll.index(en)==0):
        Data=en_data
    else:
        Data=pd.concat((Data,en_data),axis=0)


# In[ ]:


Data.to_csv(save_path,index=False)


# In[ ]:




