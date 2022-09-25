#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import datetime
from dateutil.parser import parse

#替换字符串中的匹配项
import re


# In[2]:


def process_not_timeinfo(data):
    #删除action为create_thread,delete thead所在行
    print(data.shape[0])
    data=data.drop(data[(data['action']=='create_thread') | (data['action']=='create_thread')].index)
    print(data.shape[0])
    data.reset_index(drop=True,inplace=True)
    
    #替换属性值：
    action_dict={'click_about':'navigate','click_info':'navigate',
                 'seek_video':'video','play_video':'video','load_video':'video','pause_video':'video',
                 'click_progress':'video','stop_video':'video',
                 'click_courseware':'courseware','close_courseware':'courseware',
                 'click_forum':'forum','create_comment':'forum','delete_forum':'forum','close_forum':'forum',
                 'problem_get':'problem','problem_check':'problem','problem_check_correct':'problem',
                 'problem_check_incorrect':'problem','problem_save':'problem','reset_problem':'problem'}
    #navigate
    data['action']=data['action'].map(action_dict)    #video
    #sub(模式串，（去替换模式串的字符串），主串)
    #data['action']=data['action'].map(lambda x:re.sub('video','video','pause_video'))
    
    return data


# In[11]:


def process_timeinfo(data):
    #按照日期排序
    #print("sort:")
    data=data.sort_values(by='time',ascending=True)
    new_data=pd.DataFrame(columns=['enroll_id', 'username','course_id','session_id',
                                   'action','start_time','time_span'])
    
    last_index=0
    for i,row in data.iterrows(): 
        if new_data.empty:
            print('empty!')
            stu_dict={'enroll_id':row.enroll_id,'username':row.username,
                      'course_id':row.course_id,'session_id':row.session_id,
                      'action':row.action,
                      'start_time':row.time,'time_span':0.0} 
            #设置index不重复出现
            #new_data=new_data.append(stu_dict,ignore_index=True)
            new_data=new_data.append(stu_dict,ignore_index=True)
        else:
            if (row.session_id!=new_data.loc[last_index].session_id)or(row.action!=new_data.loc[last_index].action):
                start_time=parse(new_data['start_time'][last_index])
                end_time=parse(row.time)
                new_data.loc[last_index,'time_span']=float((end_time-start_time).total_seconds())/60
                stu_dict={'enroll_id':row.enroll_id,'username':row.username,
                      'course_id':row.course_id,'session_id':row.session_id,
                      'action':row.action,
                      'start_time':row.time,'time_span':0.0} 
                new_data=new_data.append(stu_dict,ignore_index=True)
            else:
                start_time=parse(new_data['start_time'][last_index])
                end_time=parse(row.time)
                new_data.loc[last_index,'time_span']=float((end_time-start_time).total_seconds())/60#时间段按分钟存储
        #获取新dataframe的最后一个索引值
        last_index=new_data.index.values[-1]
              
    new_data.rename(columns={'username':'user_id'},inplace=True)
    return new_data


# In[4]:


read_path1='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\train_log.csv'
read_path2='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\test_log.csv'

#save_dir='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup'
save_dir="C:\\Users\\Administrator\\Desktop\\t"
save_path='C:\\Users\\Administrator\\Desktop\\1.csv'


# In[5]:


train_data=pd.read_csv(read_path1,usecols=['enroll_id', 'username','course_id','session_id','action','time'],nrows=300)
test_data=pd.read_csv(read_path2,usecols=['enroll_id', 'username','course_id','session_id','action','time'],nrows=300)


# In[6]:


data=pd.concat((train_data,test_data),axis=0)
#重置索引
data.reset_index(drop=True,inplace=True)
data=process_not_timeinfo(data)


# In[8]:


data.to_csv(save_path,index=False)


# In[9]:


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#创建文件


# In[12]:


groups=data.groupby(data['enroll_id'])
list_enroll=data['enroll_id'].drop_duplicates().dropna().to_list()
en_data=pd.DataFrame()
for en in list_enroll:
    en_data=groups.get_group(en)
    #重置索引必须inplace=True,drop=True去掉index列
    en_data.reset_index(drop=True,inplace=True)
    new_en_data=process_timeinfo(en_data)
    if (new_en_data.isnull().values.any()):
        print(en,"nan")
        print(new_en_data)
    new_en_data.to_csv(os.path.join(save_dir,str(en)+'.csv'),index=False)


# In[ ]:




