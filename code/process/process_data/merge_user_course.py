#2021-11-19

import pandas as pd
import numpy as np
import os

###preprocess stdent info
read_user_path = 'F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\user_info.csv'
#save_path = 'F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info_new.csv'
save_dir='F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info'
user_info=pd.read_csv(read_user_path)
#处理
gens=['male','female']
edus = ["Bachelor's", "High", "Master's", "Primary", "Middle", "Associate", "Doctorate"]
#gender
#要加return x 否则返回none！！！
def convert_gender(x):
    if x in gens:
        x= gens.index(x) + 1
        return x
    x=0
    return x
#enducation
def convert_edu(x):
    if x in edus:
        x=edus.index(x)+1
        return x
    x=0
    return x
#age
def convert_age(x):
    if np.isnan(x)==False:
        x = 2021 - x
        return x
    x=0
    return x
#修改列名
#必须,inplace=True
user_info.rename(columns={'birth':'age'},inplace=True)

#print(user_info)
#转换每一列
user_info['gender']=user_info['gender'].apply(lambda x:convert_gender(x))
user_info['education']=user_info['education'].apply(lambda x:convert_edu(x))
user_info['age']=user_info['age'].apply(lambda x:convert_age(x))

#process log
read_train_path='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\train_log.csv'
read_test_path='F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\prediction_log\\test_log.csv'

log_train_data=pd.read_csv(read_train_path,usecols=['enroll_id', 'username','course_id'])
log_test_data=pd.read_csv(read_test_path,usecols=['enroll_id', 'username','course_id'])

#若ignore_index=False,会出现一列索引列
log_data=pd.concat((log_train_data,log_test_data),axis=0)
#修改列名
log_data.rename(columns={'username':'user_id'},inplace=True)

#merge_user_info
data=pd.merge(user_info,log_data,on='user_id')
#(data.shape)
#keep=last:保留最后值
data.drop_duplicates(['user_id','enroll_id','course_id'],keep='last',inplace=True,ignore_index=True)

#是创建目录，而不是创建文件
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(data)
#保存文件
data.to_csv(os.path.join(save_dir,'user_log_info.csv'),index=False)
