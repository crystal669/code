#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import os

def convert_course_type(x,list_type):
    ty = 0
    if x in list_type:
        ty = list_type.index(x)+1
    return ty

def convert_course_category(x,list_category):
    cat = 0
    if x in list_category:
        cat = list_category.index(x)+1
    return cat


# In[35]:



read_course_path='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info\\course_info.csv'
save_dir='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info'


# In[36]:


course_data = pd.read_csv(read_course_path)

course_type = course_data['course_type']
course_category = course_data['category']
# 序列去重
#course_type.drop_duplicates()
list_type = course_type.drop_duplicates().dropna().tolist()
list_category = course_category.drop_duplicates().dropna().tolist()


#对每个属性进行转换
course_data['course_type'] = course_data['course_type'].apply(lambda x: convert_course_type(x,list_type))
course_data['category'] = course_data['category'].apply(lambda x: convert_course_category(x,list_category))

new_data=course_data.loc[:, ['course_id', 'course_type', 'category']]


# In[37]:


#get type num
print(len(course_type))
print(len(list_type))
# get category num
print(len(course_category ))
print(list_category)


# In[39]:


#是创建目录，而不是创建文件
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#保存文件
new_data.to_csv(os.path.join(save_dir, 'process_course_info.csv'), index=False)


# In[ ]:




