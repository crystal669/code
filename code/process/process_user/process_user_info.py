#preprocess stdent info
#-----2021-11-18---------

import pandas as pd
import numpy as np
import os

read_path = 'F:\\DeepLearning\\DataSet\\mydataset\\raw\\kddcup15\\kddcup15\\user_info.csv'
#save_path = 'F:\\DeepLearning\\DataSet\\mydataset\\process\\kddcup\\user_info_new.csv'
save_dir='C:\\Users\\Administrator\\Desktop'
user_info=pd.read_csv(read_path,nrows=400)
#将不同列转为字典
#user_gender=user_info['gender'].to_dict()
#user_education=user_info['education'].to_dict()
#user_age=user_info['birth'].to_dict()
#print(user_gender,user_education,user_age)
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

print(user_info)
#转换每一列
user_info['gender']=user_info['gender'].apply(lambda x:convert_gender(x))
print(user_info['gender'])
user_info['education']=user_info['education'].apply(lambda x:convert_edu(x))
user_info['age']=user_info['age'].apply(lambda x:convert_age(x))

print(user_info)

#是创建目录，而不是创建文件
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#保存文件
user_info.to_csv(os.path.join(save_dir,'convert_user_info.csv'),index=False)